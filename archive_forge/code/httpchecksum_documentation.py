import base64
import io
import logging
from binascii import crc32
from hashlib import sha1, sha256
from botocore.compat import HAS_CRT
from botocore.exceptions import (
from botocore.response import StreamingBody
from botocore.utils import (
 The interfaces in this module are not intended for public use.

This module defines interfaces for applying checksums to HTTP requests within
the context of botocore. This involves both resolving the checksum to be used
based on client configuration and environment, as well as application of the
checksum to the request.
