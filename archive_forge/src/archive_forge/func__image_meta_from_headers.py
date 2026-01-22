import copy
from oslo_utils import encodeutils
from oslo_utils import strutils
import urllib.parse
from glanceclient.common import utils
from glanceclient.v1.apiclient import base
def _image_meta_from_headers(self, headers):
    meta = {'properties': {}}
    safe_decode = encodeutils.safe_decode
    for key, value in headers.items():
        key = key.lower()
        value = safe_decode(value, incoming='utf-8')
        if key.startswith('x-image-meta-property-'):
            _key = safe_decode(key[22:], incoming='utf-8')
            meta['properties'][_key] = value
        elif key.startswith('x-image-meta-'):
            _key = safe_decode(key[13:], incoming='utf-8')
            meta[_key] = value
    for key in ['is_public', 'protected', 'deleted']:
        if key in meta:
            meta[key] = strutils.bool_from_string(meta[key])
    return self._format_image_meta_for_user(meta)