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