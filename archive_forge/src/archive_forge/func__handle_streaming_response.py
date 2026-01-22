import base64
import io
import logging
from binascii import crc32
from hashlib import sha1, sha256
from botocore.compat import HAS_CRT
from botocore.exceptions import (
from botocore.response import StreamingBody
from botocore.utils import (
def _handle_streaming_response(http_response, response, algorithm):
    checksum_cls = _CHECKSUM_CLS.get(algorithm)
    header_name = 'x-amz-checksum-%s' % algorithm
    return StreamingChecksumBody(http_response.raw, response['headers'].get('content-length'), checksum_cls(), response['headers'][header_name])