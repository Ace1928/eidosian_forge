from calendar import timegm
import collections
from hashlib import sha1
import hmac
import json
import os
import time
from urllib import parse
from openstack import _log
from openstack.cloud import _utils
from openstack import exceptions
from openstack.object_store.v1 import account as _account
from openstack.object_store.v1 import container as _container
from openstack.object_store.v1 import info as _info
from openstack.object_store.v1 import obj as _obj
from openstack import proxy
from openstack import utils
def _upload_large_object(self, endpoint, filename, headers, file_size, segment_size, use_slo):
    segment_futures = []
    segment_results = []
    retry_results = []
    retry_futures = []
    manifest = []
    segments = self._get_file_segments(endpoint, filename, file_size, segment_size)
    for name, segment in segments.items():
        segment_future = self._connection._pool_executor.submit(self.put, name, headers=headers, data=segment, raise_exc=False)
        segment_futures.append(segment_future)
        manifest.append(dict(path='/{name}'.format(name=parse.unquote(name)), size_bytes=segment.length))
    segment_results, retry_results = self._connection._wait_for_futures(segment_futures, raise_on_error=False)
    self._add_etag_to_manifest(segment_results, manifest)
    for result in retry_results:
        name = self._object_name_from_url(result.url)
        segment = segments[name]
        segment.seek(0)
        segment_future = self._connection._pool_executor.submit(self.put, name, headers=headers, data=segment)
        retry_futures.append(segment_future)
    segment_results, retry_results = self._connection._wait_for_futures(retry_futures, raise_on_error=True)
    self._add_etag_to_manifest(segment_results, manifest)
    try:
        if use_slo:
            return self._finish_large_object_slo(endpoint, headers, manifest)
        else:
            return self._finish_large_object_dlo(endpoint, headers)
    except Exception:
        try:
            segment_prefix = endpoint.split('/')[-1]
            self.log.debug('Failed to upload large object manifest for %s. Removing segment uploads.', segment_prefix)
            self._delete_autocreated_image_objects(segment_prefix=segment_prefix)
        except Exception:
            self.log.exception('Failed to cleanup image objects for %s:', segment_prefix)
        raise