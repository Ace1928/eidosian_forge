import warnings
import base64
import typing as t
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import UnsupportedAlgorithm
from requests.auth import AuthBase
from requests.packages.urllib3.response import HTTPResponse
import spnego
class HttpNtlmAuth(AuthBase):
    """
    HTTP NTLM Authentication Handler for Requests.

    Supports pass-the-hash.
    """

    def __init__(self, username, password, session=None, send_cbt=True):
        """Create an authentication handler for NTLM over HTTP.

        :param str username: Username in 'domain\\username' format
        :param str password: Password
        :param str session: Unused. Kept for backwards-compatibility.
        :param bool send_cbt: Will send the channel bindings over a HTTPS channel (Default: True)
        """
        self.username = username
        self.password = password
        self.send_cbt = send_cbt
        self.session_security = None

    def retry_using_http_NTLM_auth(self, auth_header_field, auth_header, response, auth_type, args):
        server_certificate_hash = self._get_server_cert(response)
        cbt = None
        if server_certificate_hash:
            cbt = spnego.channel_bindings.GssChannelBindings(application_data=b'tls-server-end-point:' + server_certificate_hash)
        'Attempt to authenticate using HTTP NTLM challenge/response.'
        if auth_header in response.request.headers:
            return response
        content_length = int(response.request.headers.get('Content-Length', '0'), base=10)
        if hasattr(response.request.body, 'seek'):
            if content_length > 0:
                response.request.body.seek(-content_length, 1)
            else:
                response.request.body.seek(0, 0)
        response.content
        response.raw.release_conn()
        request = response.request.copy()
        client = spnego.client(self.username, self.password, protocol='ntlm', channel_bindings=cbt)
        negotiate_message = base64.b64encode(client.step()).decode()
        auth = '%s %s' % (auth_type, negotiate_message)
        request.headers[auth_header] = auth
        args_nostream = dict(args, stream=False)
        response2 = response.connection.send(request, **args_nostream)
        response2.content
        response2.raw.release_conn()
        request = response2.request.copy()
        if response2.headers.get('set-cookie'):
            request.headers['Cookie'] = response2.headers.get('set-cookie')
        auth_header_value = response2.headers[auth_header_field]
        auth_strip = auth_type + ' '
        ntlm_header_value = next((s for s in (val.lstrip() for val in auth_header_value.split(',')) if s.startswith(auth_strip))).strip()
        val = base64.b64decode(ntlm_header_value[len(auth_strip):].encode())
        authenticate_message = base64.b64encode(client.step(val)).decode()
        auth = '%s %s' % (auth_type, authenticate_message)
        request.headers[auth_header] = auth
        response3 = response2.connection.send(request, **args)
        response3.history.append(response)
        response3.history.append(response2)
        self.session_security = ShimSessionSecurity(client)
        return response3

    def response_hook(self, r, **kwargs):
        """The actual hook handler."""
        if r.status_code == 401:
            www_authenticate = r.headers.get('www-authenticate', '').lower()
            auth_type = _auth_type_from_header(www_authenticate)
            if auth_type is not None:
                return self.retry_using_http_NTLM_auth('www-authenticate', 'Authorization', r, auth_type, kwargs)
        elif r.status_code == 407:
            proxy_authenticate = r.headers.get('proxy-authenticate', '').lower()
            auth_type = _auth_type_from_header(proxy_authenticate)
            if auth_type is not None:
                return self.retry_using_http_NTLM_auth('proxy-authenticate', 'Proxy-authorization', r, auth_type, kwargs)
        return r

    def _get_server_cert(self, response):
        """
        Get the certificate at the request_url and return it as a hash. Will get the raw socket from the
        original response from the server. This socket is then checked if it is an SSL socket and then used to
        get the hash of the certificate. The certificate hash is then used with NTLMv2 authentication for
        Channel Binding Tokens support. If the raw object is not a urllib3 HTTPReponse (default with requests)
        then no certificate will be returned.

        :param response: The original 401 response from the server
        :return: The hash of the DER encoded certificate at the request_url or None if not a HTTPS endpoint
        """
        if self.send_cbt:
            raw_response = response.raw
            if isinstance(raw_response, HTTPResponse):
                socket = raw_response._fp.fp.raw._sock
                try:
                    server_certificate = socket.getpeercert(True)
                except AttributeError:
                    pass
                else:
                    return _get_certificate_hash(server_certificate)
            else:
                warnings.warn('Requests is running with a non urllib3 backend, cannot retrieve server certificate for CBT', NoCertificateRetrievedWarning)
        return None

    def __call__(self, r):
        r.headers['Connection'] = 'Keep-Alive'
        r.register_hook('response', self.response_hook)
        return r