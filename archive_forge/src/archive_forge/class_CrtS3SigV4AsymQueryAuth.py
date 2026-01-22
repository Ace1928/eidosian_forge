import datetime
from io import BytesIO
from botocore.auth import (
from botocore.compat import HTTPHeaders, awscrt, parse_qs, urlsplit, urlunsplit
from botocore.exceptions import NoCredentialsError
from botocore.utils import percent_encode_sequence
class CrtS3SigV4AsymQueryAuth(CrtSigV4AsymQueryAuth):
    """S3 SigV4A auth using query parameters.
    This signer will sign a request using query parameters and signature
    version 4A, i.e a "presigned url" signer.
    """
    _USE_DOUBLE_URI_ENCODE = False
    _SHOULD_NORMALIZE_URI_PATH = False

    def _should_sha256_sign_payload(self, request):
        return False

    def _should_add_content_sha256_header(self, explicit_payload):
        return False