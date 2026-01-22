import json
import logging as log
from urllib import parse as urlparse
import netaddr
from oslo_concurrency.lockutils import synchronized
import requests
from osprofiler.drivers import base
from osprofiler import exc
class LogInsightClient(object):
    """A minimal Log Insight client."""
    LI_OSPROFILER_AGENT_ID = 'F52D775B-6017-4787-8C8A-F21AE0AEC057'
    SESSIONS_PATH = 'api/v1/sessions'
    CURRENT_SESSIONS_PATH = 'api/v1/sessions/current'
    EVENTS_INGEST_PATH = 'api/v1/events/ingest/%s' % LI_OSPROFILER_AGENT_ID
    QUERY_EVENTS_BASE_PATH = 'api/v1/events'

    def __init__(self, host, username, password, api_port=9000, api_ssl_port=9543, query_timeout=60000):
        self._host = host
        self._username = username
        self._password = password
        self._api_port = api_port
        self._api_ssl_port = api_ssl_port
        self._query_timeout = query_timeout
        self._session = requests.Session()
        self._session_id = None

    def _build_base_url(self, scheme):
        proto_str = '%s://' % scheme
        host_str = '[%s]' % self._host if netaddr.valid_ipv6(self._host) else self._host
        port_str = ':%d' % (self._api_ssl_port if scheme == 'https' else self._api_port)
        return proto_str + host_str + port_str

    def _check_response(self, resp):
        if resp.status_code == 440:
            raise exc.LogInsightLoginTimeout()
        if not resp.ok:
            msg = 'n/a'
            if resp.text:
                try:
                    body = json.loads(resp.text)
                    msg = body.get('errorMessage', msg)
                except ValueError:
                    pass
            else:
                msg = resp.reason
            raise exc.LogInsightAPIError(msg)

    def _send_request(self, method, scheme, path, headers=None, body=None, params=None):
        url = '%s/%s' % (self._build_base_url(scheme), path)
        headers = headers or {}
        headers['content-type'] = 'application/json'
        body = body or {}
        params = params or {}
        req = requests.Request(method, url, headers=headers, data=json.dumps(body), params=params)
        req = req.prepare()
        resp = self._session.send(req, verify=False)
        self._check_response(resp)
        return resp.json()

    def _get_auth_header(self):
        return {'X-LI-Session-Id': self._session_id}

    def _trunc_session_id(self):
        if self._session_id:
            return self._session_id[-5:]

    def _is_current_session_active(self):
        try:
            self._send_request('get', 'https', self.CURRENT_SESSIONS_PATH, headers=self._get_auth_header())
            LOG.debug('Current session %s is active.', self._trunc_session_id())
            return True
        except (exc.LogInsightLoginTimeout, exc.LogInsightAPIError):
            LOG.debug('Current session %s is not active.', self._trunc_session_id())
            return False

    @synchronized('li_login_lock')
    def login(self):
        if self._session_id and self._is_current_session_active():
            return
        LOG.info('Logging into Log Insight server: %s.', self._host)
        resp = self._send_request('post', 'https', self.SESSIONS_PATH, body={'username': self._username, 'password': self._password})
        self._session_id = resp['sessionId']
        LOG.debug('Established session %s.', self._trunc_session_id())

    def send_event(self, event):
        events = {'events': [event]}
        self._send_request('post', 'http', self.EVENTS_INGEST_PATH, body=events)

    def query_events(self, params):
        constraints = []
        for field, value in params.items():
            constraints.append('%s/CONTAINS+%s' % (field, value))
        constraints.append('timestamp/GT+0')
        path = '%s/%s' % (self.QUERY_EVENTS_BASE_PATH, '/'.join(constraints))

        def _query_events():
            return self._send_request('get', 'https', path, headers=self._get_auth_header(), params={'limit': 20000, 'timeout': self._query_timeout})
        try:
            resp = _query_events()
        except exc.LogInsightLoginTimeout:
            LOG.debug('Current session timed out.')
            self.login()
            resp = _query_events()
        return resp