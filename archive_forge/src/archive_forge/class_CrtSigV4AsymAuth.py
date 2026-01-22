import datetime
from io import BytesIO
from botocore.auth import (
from botocore.compat import HTTPHeaders, awscrt, parse_qs, urlsplit, urlunsplit
from botocore.exceptions import NoCredentialsError
from botocore.utils import percent_encode_sequence
class CrtSigV4AsymAuth(BaseSigner):
    REQUIRES_REGION = True
    _PRESIGNED_HEADERS_BLOCKLIST = ['Authorization', 'X-Amz-Date', 'X-Amz-Content-SHA256', 'X-Amz-Security-Token']
    _SIGNATURE_TYPE = awscrt.auth.AwsSignatureType.HTTP_REQUEST_HEADERS
    _USE_DOUBLE_URI_ENCODE = True
    _SHOULD_NORMALIZE_URI_PATH = True

    def __init__(self, credentials, service_name, region_name):
        self.credentials = credentials
        self._service_name = service_name
        self._region_name = region_name
        self._expiration_in_seconds = None

    def add_auth(self, request):
        if self.credentials is None:
            raise NoCredentialsError()
        datetime_now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
        existing_sha256 = self._get_existing_sha256(request)
        self._modify_request_before_signing(request)
        credentials_provider = awscrt.auth.AwsCredentialsProvider.new_static(access_key_id=self.credentials.access_key, secret_access_key=self.credentials.secret_key, session_token=self.credentials.token)
        if self._is_streaming_checksum_payload(request):
            explicit_payload = STREAMING_UNSIGNED_PAYLOAD_TRAILER
        elif self._should_sha256_sign_payload(request):
            if existing_sha256:
                explicit_payload = existing_sha256
            else:
                explicit_payload = None
        else:
            explicit_payload = UNSIGNED_PAYLOAD
        if self._should_add_content_sha256_header(explicit_payload):
            body_header = awscrt.auth.AwsSignedBodyHeaderType.X_AMZ_CONTENT_SHA_256
        else:
            body_header = awscrt.auth.AwsSignedBodyHeaderType.NONE
        signing_config = awscrt.auth.AwsSigningConfig(algorithm=awscrt.auth.AwsSigningAlgorithm.V4_ASYMMETRIC, signature_type=self._SIGNATURE_TYPE, credentials_provider=credentials_provider, region=self._region_name, service=self._service_name, date=datetime_now, should_sign_header=self._should_sign_header, use_double_uri_encode=self._USE_DOUBLE_URI_ENCODE, should_normalize_uri_path=self._SHOULD_NORMALIZE_URI_PATH, signed_body_value=explicit_payload, signed_body_header_type=body_header, expiration_in_seconds=self._expiration_in_seconds)
        crt_request = self._crt_request_from_aws_request(request)
        future = awscrt.auth.aws_sign_request(crt_request, signing_config)
        future.result()
        self._apply_signing_changes(request, crt_request)

    def _crt_request_from_aws_request(self, aws_request):
        url_parts = urlsplit(aws_request.url)
        crt_path = url_parts.path if url_parts.path else '/'
        if aws_request.params:
            array = []
            for param, value in aws_request.params.items():
                value = str(value)
                array.append(f'{param}={value}')
            crt_path = crt_path + '?' + '&'.join(array)
        elif url_parts.query:
            crt_path = f'{crt_path}?{url_parts.query}'
        crt_headers = awscrt.http.HttpHeaders(aws_request.headers.items())
        crt_body_stream = None
        if aws_request.body:
            if hasattr(aws_request.body, 'seek'):
                crt_body_stream = aws_request.body
            else:
                crt_body_stream = BytesIO(aws_request.body)
        crt_request = awscrt.http.HttpRequest(method=aws_request.method, path=crt_path, headers=crt_headers, body_stream=crt_body_stream)
        return crt_request

    def _apply_signing_changes(self, aws_request, signed_crt_request):
        aws_request.headers = HTTPHeaders.from_pairs(list(signed_crt_request.headers))

    def _should_sign_header(self, name, **kwargs):
        return name.lower() not in SIGNED_HEADERS_BLACKLIST

    def _modify_request_before_signing(self, request):
        for h in self._PRESIGNED_HEADERS_BLOCKLIST:
            if h in request.headers:
                del request.headers[h]
        if 'host' not in request.headers:
            request.headers['host'] = _host_from_url(request.url)

    def _get_existing_sha256(self, request):
        return request.headers.get('X-Amz-Content-SHA256')

    def _is_streaming_checksum_payload(self, request):
        checksum_context = request.context.get('checksum', {})
        algorithm = checksum_context.get('request_algorithm')
        return isinstance(algorithm, dict) and algorithm.get('in') == 'trailer'

    def _should_sha256_sign_payload(self, request):
        if not request.url.startswith('https'):
            return True
        return request.context.get('payload_signing_enabled', True)

    def _should_add_content_sha256_header(self, explicit_payload):
        return explicit_payload is not None