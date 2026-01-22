import datetime
from io import BytesIO
from botocore.auth import (
from botocore.compat import HTTPHeaders, awscrt, parse_qs, urlsplit, urlunsplit
from botocore.exceptions import NoCredentialsError
from botocore.utils import percent_encode_sequence
def _is_streaming_checksum_payload(self, request):
    checksum_context = request.context.get('checksum', {})
    algorithm = checksum_context.get('request_algorithm')
    return isinstance(algorithm, dict) and algorithm.get('in') == 'trailer'