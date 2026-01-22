import httplib2
import logging
import os
import sys
import time
from troveclient.compat import auth
from troveclient.compat import exceptions
class TroveHTTPClient(httplib2.Http):
    USER_AGENT = 'python-troveclient'

    def __init__(self, user, password, tenant, auth_url, service_name, service_url=None, auth_strategy=None, insecure=False, timeout=None, proxy_tenant_id=None, proxy_token=None, region_name=None, endpoint_type='publicURL', service_type=None, timings=False):
        super(TroveHTTPClient, self).__init__(timeout=timeout)
        self.username = user
        self.password = password
        self.tenant = tenant
        if auth_url:
            self.auth_url = auth_url.rstrip('/')
        else:
            self.auth_url = None
        self.region_name = region_name
        self.endpoint_type = endpoint_type
        self.service_url = service_url
        self.service_type = service_type
        self.service_name = service_name
        self.timings = timings
        self.times = []
        self.auth_token = None
        self.proxy_token = proxy_token
        self.proxy_tenant_id = proxy_tenant_id
        self.force_exception_to_status_code = True
        self.disable_ssl_certificate_validation = insecure
        auth_cls = auth.get_authenticator_cls(auth_strategy)
        self.authenticator = auth_cls(self, auth_strategy, self.auth_url, self.username, self.password, self.tenant, region=region_name, service_type=service_type, service_name=service_name, service_url=service_url)
        if hasattr(self.authenticator, 'auth'):
            self.auth = self.authenticator.auth

    def get_timings(self):
        return self.times

    def http_log(self, args, kwargs, resp, body):
        if not RDC_PP:
            self.simple_log(args, kwargs, resp, body)
        else:
            self.pretty_log(args, kwargs, resp, body)

    def simple_log(self, args, kwargs, resp, body):
        if not LOG.isEnabledFor(logging.DEBUG):
            return
        string_parts = ['curl -i']
        for element in args:
            if element in ('GET', 'POST'):
                string_parts.append(' -X %s' % element)
            else:
                string_parts.append(' %s' % element)
        for element in kwargs['headers']:
            header = ' -H "%s: %s"' % (element, kwargs['headers'][element])
            string_parts.append(header)
        LOG.debug('REQ: %s\n', ''.join(string_parts))
        if 'body' in kwargs:
            LOG.debug('REQ BODY: %s\n', kwargs['body'])
        LOG.debug('RESP:%s %s\n', resp, body)

    def pretty_log(self, args, kwargs, resp, body):
        if not LOG.isEnabledFor(logging.DEBUG):
            return
        string_parts = ['curl -i']
        for element in args:
            if element in ('GET', 'POST'):
                string_parts.append(' -X %s' % element)
            else:
                string_parts.append(' %s' % element)
        for element in kwargs['headers']:
            header = ' -H "%s: %s"' % (element, kwargs['headers'][element])
            string_parts.append(header)
        curl_cmd = ''.join(string_parts)
        LOG.debug('REQUEST:')
        if 'body' in kwargs:
            LOG.debug("%s -d '%s'", curl_cmd, kwargs['body'])
            try:
                req_body = json.dumps(json.loads(kwargs['body']), sort_keys=True, indent=4)
            except Exception:
                req_body = kwargs['body']
            LOG.debug('BODY: %s\n', req_body)
        else:
            LOG.debug(curl_cmd)
        try:
            resp_body = json.dumps(json.loads(body), sort_keys=True, indent=4)
        except Exception:
            resp_body = body
        LOG.debug('RESPONSE HEADERS: %s', resp)
        LOG.debug('RESPONSE BODY   : %s', resp_body)

    def request(self, *args, **kwargs):
        kwargs.setdefault('headers', kwargs.get('headers', {}))
        kwargs['headers']['User-Agent'] = self.USER_AGENT
        self.morph_request(kwargs)
        resp, body = super(TroveHTTPClient, self).request(*args, **kwargs)
        resp.status_code = resp.status
        self.last_response = (resp, body)
        self.http_log(args, kwargs, resp, body)
        if body:
            try:
                body = self.morph_response_body(body)
            except exceptions.ResponseFormatError:
                self.raise_error_from_status(resp, None)
                raise
        else:
            body = None
        if resp.status in expected_errors:
            raise exceptions.from_response(resp, body)
        return (resp, body)

    def raise_error_from_status(self, resp, body):
        if resp.status in expected_errors:
            raise exceptions.from_response(resp, body)

    def morph_request(self, kwargs):
        kwargs['headers']['Accept'] = 'application/json'
        kwargs['headers']['Content-Type'] = 'application/json'
        if 'body' in kwargs:
            kwargs['body'] = json.dumps(kwargs['body'])

    def morph_response_body(self, raw_body):
        try:
            return json.loads(raw_body.decode())
        except ValueError:
            raise exceptions.ResponseFormatError()

    def _time_request(self, url, method, **kwargs):
        start_time = time.time()
        resp, body = self.request(url, method, **kwargs)
        self.times.append(('%s %s' % (method, url), start_time, time.time()))
        return (resp, body)

    def _cs_request(self, url, method, **kwargs):

        def request():
            kwargs.setdefault('headers', {})['X-Auth-Token'] = self.auth_token
            if self.tenant:
                kwargs['headers']['X-Auth-Project-Id'] = self.tenant
            resp, body = self._time_request(self.service_url + url, method, **kwargs)
            return (resp, body)
        if not self.auth_token or not self.service_url:
            self.authenticate()
        try:
            return request()
        except exceptions.Unauthorized:
            self.authenticate()
            return request()

    def get(self, url, **kwargs):
        return self._cs_request(url, 'GET', **kwargs)

    def patch(self, url, **kwargs):
        return self._cs_request(url, 'PATCH', **kwargs)

    def post(self, url, **kwargs):
        return self._cs_request(url, 'POST', **kwargs)

    def put(self, url, **kwargs):
        return self._cs_request(url, 'PUT', **kwargs)

    def delete(self, url, **kwargs):
        return self._cs_request(url, 'DELETE', **kwargs)

    def authenticate(self):
        """Auths the client and gets a token. May optionally set a service url.

        The client will get auth errors until the authentication step
        occurs. Additionally, if a service_url was not explicitly given in
        the clients __init__ method, one will be obtained from the auth
        service.

        """
        catalog = self.authenticator.authenticate()
        if self.service_url:
            possible_service_url = None
        elif self.endpoint_type == 'publicURL':
            possible_service_url = catalog.get_public_url()
        elif self.endpoint_type == 'adminURL':
            possible_service_url = catalog.get_management_url()
        self.authenticate_with_token(catalog.get_token(), possible_service_url)

    def authenticate_with_token(self, token, service_url=None):
        self.auth_token = token
        if not self.service_url:
            if not service_url:
                raise exceptions.ServiceUrlNotGiven()
            else:
                self.service_url = service_url