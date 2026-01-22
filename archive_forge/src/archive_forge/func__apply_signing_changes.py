import datetime
from io import BytesIO
from botocore.auth import (
from botocore.compat import HTTPHeaders, awscrt, parse_qs, urlsplit, urlunsplit
from botocore.exceptions import NoCredentialsError
from botocore.utils import percent_encode_sequence
def _apply_signing_changes(self, aws_request, signed_crt_request):
    super()._apply_signing_changes(aws_request, signed_crt_request)
    signed_query = urlsplit(signed_crt_request.path).query
    p = urlsplit(aws_request.url)
    aws_request.url = urlunsplit((p[0], p[1], p[2], signed_query, p[4]))