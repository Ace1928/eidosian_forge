from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import re
import threading
import time
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.logging import common
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.credentials import requests as creds_requests
from googlecloudsdk.core.util import encoding
import requests
class GCSLogTailer(TailerBase):
    """Helper class to tail a GCS logfile, printing content as available."""
    LOG_OUTPUT_INCOMPLETE = ' (possibly incomplete) '
    GCS_URL_PATTERN = 'https://www.googleapis.com/storage/v1/b/{bucket}/o/{obj}?alt=media'

    def __init__(self, bucket, obj, out=log.status, url_pattern=None):
        self.transport = RequestsLogTailer()
        url_pattern = url_pattern or self.GCS_URL_PATTERN
        self.url = url_pattern.format(bucket=bucket, obj=obj)
        log.debug('GCS logfile url is ' + self.url)
        self.cursor = 0
        self.out = out
        self.stop = False

    @classmethod
    def FromBuild(cls, build, out=log.out):
        """Build a GCSLogTailer from a build resource.

    Args:
      build: Build resource, The build whose logs shall be streamed.
      out: The output stream to write the logs to.

    Raises:
      NoLogsBucketException: If the build does not specify a logsBucket.

    Returns:
      GCSLogTailer, the tailer of this build's logs.
    """
        if not build.logsBucket:
            raise NoLogsBucketException()
        log_stripped = build.logsBucket
        gcs_prefix = 'gs://'
        if log_stripped.startswith(gcs_prefix):
            log_stripped = log_stripped[len(gcs_prefix):]
        if '/' not in log_stripped:
            log_bucket = log_stripped
            log_object_dir = ''
        else:
            [log_bucket, log_object_dir] = log_stripped.split('/', 1)
            log_object_dir += '/'
        log_object = '{object}log-{id}.txt'.format(object=log_object_dir, id=build.id)
        return cls(bucket=log_bucket, obj=log_object, out=out, url_pattern='https://storage.googleapis.com/{bucket}/{obj}')

    def Poll(self, is_last=False):
        """Poll the GCS object and print any new bytes to the console.

    Args:
      is_last: True if this is the final poll operation.

    Raises:
      api_exceptions.HttpError: if there is trouble connecting to GCS.
      api_exceptions.CommunicationError: if there is trouble reaching the server
          and is_last=True.
    """
        try:
            res = self.transport.Request(self.url, self.cursor)
        except api_exceptions.CommunicationError:
            if is_last:
                raise
            return
        if res.status == 404:
            log.debug('Reading GCS logfile: 404 (no log yet; keep polling)')
            return
        if res.status == 416:
            log.debug('Reading GCS logfile: 416 (no new content; keep polling)')
            if is_last:
                self._PrintLastLine()
            return
        if res.status == 206 or res.status == 200:
            log.debug('Reading GCS logfile: {code} (read {count} bytes)'.format(code=res.status, count=len(res.body)))
            if self.cursor == 0:
                self._PrintFirstLine()
            self.cursor += len(res.body)
            decoded = encoding.Decode(res.body)
            if decoded is not None:
                decoded = self._ValidateScreenReader(decoded)
            self._PrintLogLine(decoded.rstrip('\n'))
            if is_last:
                self._PrintLastLine()
            return
        if res.status == 429:
            log.warning('Reading GCS logfile: 429 (server is throttling us)')
            if is_last:
                self._PrintLastLine(self.LOG_OUTPUT_INCOMPLETE)
            return
        if res.status >= 500 and res.status < 600:
            log.warning('Reading GCS logfile: got {0}, retrying'.format(res.status))
            if is_last:
                self._PrintLastLine(self.LOG_OUTPUT_INCOMPLETE)
            return
        headers = dict(res.headers)
        headers['status'] = res.status
        raise api_exceptions.HttpError(headers, res.body, self.url)

    def Tail(self):
        """Tail the GCS object and print any new bytes to the console."""
        while not self.stop:
            self.Poll()
            time.sleep(1)
        self.Poll(is_last=True)

    def Stop(self):
        """Stop log tailing."""
        self.stop = True

    def Print(self):
        """Print GCS logs to the console."""
        self.Poll(is_last=True)