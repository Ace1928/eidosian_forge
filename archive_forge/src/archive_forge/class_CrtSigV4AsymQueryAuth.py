import datetime
from io import BytesIO
from botocore.auth import (
from botocore.compat import HTTPHeaders, awscrt, parse_qs, urlsplit, urlunsplit
from botocore.exceptions import NoCredentialsError
from botocore.utils import percent_encode_sequence
class CrtSigV4AsymQueryAuth(CrtSigV4AsymAuth):
    DEFAULT_EXPIRES = 3600
    _SIGNATURE_TYPE = awscrt.auth.AwsSignatureType.HTTP_REQUEST_QUERY_PARAMS

    def __init__(self, credentials, service_name, region_name, expires=DEFAULT_EXPIRES):
        super().__init__(credentials, service_name, region_name)
        self._expiration_in_seconds = expires

    def _modify_request_before_signing(self, request):
        super()._modify_request_before_signing(request)
        content_type = request.headers.get('content-type')
        if content_type == 'application/x-www-form-urlencoded; charset=utf-8':
            del request.headers['content-type']
        url_parts = urlsplit(request.url)
        query_string_parts = parse_qs(url_parts.query, keep_blank_values=True)
        query_dict = {k: v[0] for k, v in query_string_parts.items()}
        if request.data:
            query_dict.update(_get_body_as_dict(request))
            request.data = ''
        new_query_string = percent_encode_sequence(query_dict)
        p = url_parts
        new_url_parts = (p[0], p[1], p[2], new_query_string, p[4])
        request.url = urlunsplit(new_url_parts)

    def _apply_signing_changes(self, aws_request, signed_crt_request):
        super()._apply_signing_changes(aws_request, signed_crt_request)
        signed_query = urlsplit(signed_crt_request.path).query
        p = urlsplit(aws_request.url)
        aws_request.url = urlunsplit((p[0], p[1], p[2], signed_query, p[4]))