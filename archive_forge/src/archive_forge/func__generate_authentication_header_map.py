import hashlib
import hmac
import json
import os
import posixpath
import re
from six.moves import http_client
from six.moves import urllib
from six.moves.urllib.parse import urljoin
from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import external_account
def _generate_authentication_header_map(host, canonical_uri, canonical_querystring, method, region, access_key, secret_key, security_token, request_payload='', additional_headers={}):
    """Generates the authentication header map needed for generating the AWS
    Signature Version 4 signed request.

    Args:
        host (str): The AWS service URL hostname.
        canonical_uri (str): The AWS service URL path name.
        canonical_querystring (str): The AWS service URL query string.
        method (str): The HTTP method used to call this API.
        region (str): The AWS region.
        access_key (str): The AWS access key ID.
        secret_key (str): The AWS secret access key.
        security_token (Optional[str]): The AWS security session token. This is
            available for temporary sessions.
        request_payload (Optional[str]): The optional request payload if
            available.
        additional_headers (Optional[Mapping[str, str]]): The optional
            additional headers needed for the requested AWS API.

    Returns:
        Mapping[str, str]: The AWS authentication header dictionary object.
            This contains the x-amz-date and authorization header information.
    """
    service_name = host.split('.')[0]
    current_time = _helpers.utcnow()
    amz_date = current_time.strftime('%Y%m%dT%H%M%SZ')
    date_stamp = current_time.strftime('%Y%m%d')
    full_headers = {}
    for key in additional_headers:
        full_headers[key.lower()] = additional_headers[key]
    if security_token is not None:
        full_headers[_AWS_SECURITY_TOKEN_HEADER] = security_token
    full_headers['host'] = host
    if 'date' not in full_headers:
        full_headers[_AWS_DATE_HEADER] = amz_date
    canonical_headers = ''
    header_keys = list(full_headers.keys())
    header_keys.sort()
    for key in header_keys:
        canonical_headers = '{}{}:{}\n'.format(canonical_headers, key, full_headers[key])
    signed_headers = ';'.join(header_keys)
    payload_hash = hashlib.sha256((request_payload or '').encode('utf-8')).hexdigest()
    canonical_request = '{}\n{}\n{}\n{}\n{}\n{}'.format(method, canonical_uri, canonical_querystring, canonical_headers, signed_headers, payload_hash)
    credential_scope = '{}/{}/{}/{}'.format(date_stamp, region, service_name, _AWS_REQUEST_TYPE)
    string_to_sign = '{}\n{}\n{}\n{}'.format(_AWS_ALGORITHM, amz_date, credential_scope, hashlib.sha256(canonical_request.encode('utf-8')).hexdigest())
    signing_key = _get_signing_key(secret_key, date_stamp, region, service_name)
    signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
    authorization_header = '{} Credential={}/{}, SignedHeaders={}, Signature={}'.format(_AWS_ALGORITHM, access_key, credential_scope, signed_headers, signature)
    authentication_header = {'authorization_header': authorization_header}
    if 'date' not in full_headers:
        authentication_header['amz_date'] = amz_date
    return authentication_header