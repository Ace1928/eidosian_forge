import base64
import io
import logging
from binascii import crc32
from hashlib import sha1, sha256
from botocore.compat import HAS_CRT
from botocore.exceptions import (
from botocore.response import StreamingBody
from botocore.utils import (
def _handle_bytes_response(http_response, response, algorithm):
    body = http_response.content
    header_name = 'x-amz-checksum-%s' % algorithm
    checksum_cls = _CHECKSUM_CLS.get(algorithm)
    checksum = checksum_cls()
    checksum.update(body)
    expected = response['headers'][header_name]
    if checksum.digest() != base64.b64decode(expected):
        error_msg = 'Expected checksum %s did not match calculated checksum: %s' % (expected, checksum.b64digest())
        raise FlexibleChecksumError(error_msg=error_msg)
    return body