import weakref
import threading
import time
import re
from paramiko.common import (
from paramiko.message import Message
from paramiko.util import b, u
from paramiko.ssh_exception import (
from paramiko.server import InteractiveQuery
from paramiko.ssh_gss import GSSAuth, GSS_EXCEPTIONS
class GssapiWithMicAuthHandler:
    """A specialized Auth handler for gssapi-with-mic

    During the GSSAPI token exchange we need a modified dispatch table,
    because the packet type numbers are not unique.
    """
    method = 'gssapi-with-mic'

    def __init__(self, delegate, sshgss):
        self._delegate = delegate
        self.sshgss = sshgss

    def abort(self):
        self._restore_delegate_auth_handler()
        return self._delegate.abort()

    @property
    def transport(self):
        return self._delegate.transport

    @property
    def _send_auth_result(self):
        return self._delegate._send_auth_result

    @property
    def auth_username(self):
        return self._delegate.auth_username

    @property
    def gss_host(self):
        return self._delegate.gss_host

    def _restore_delegate_auth_handler(self):
        self.transport.auth_handler = self._delegate

    def _parse_userauth_gssapi_token(self, m):
        client_token = m.get_string()
        sshgss = self.sshgss
        try:
            token = sshgss.ssh_accept_sec_context(self.gss_host, client_token, self.auth_username)
        except Exception as e:
            self.transport.saved_exception = e
            result = AUTH_FAILED
            self._restore_delegate_auth_handler()
            self._send_auth_result(self.auth_username, self.method, result)
            raise
        if token is not None:
            m = Message()
            m.add_byte(cMSG_USERAUTH_GSSAPI_TOKEN)
            m.add_string(token)
            self.transport._expected_packet = (MSG_USERAUTH_GSSAPI_TOKEN, MSG_USERAUTH_GSSAPI_MIC, MSG_USERAUTH_REQUEST)
            self.transport._send_message(m)

    def _parse_userauth_gssapi_mic(self, m):
        mic_token = m.get_string()
        sshgss = self.sshgss
        username = self.auth_username
        self._restore_delegate_auth_handler()
        try:
            sshgss.ssh_check_mic(mic_token, self.transport.session_id, username)
        except Exception as e:
            self.transport.saved_exception = e
            result = AUTH_FAILED
            self._send_auth_result(username, self.method, result)
            raise
        result = AUTH_SUCCESSFUL
        self.transport.server_object.check_auth_gssapi_with_mic(username, result)
        self._send_auth_result(username, self.method, result)

    def _parse_service_request(self, m):
        self._restore_delegate_auth_handler()
        return self._delegate._parse_service_request(m)

    def _parse_userauth_request(self, m):
        self._restore_delegate_auth_handler()
        return self._delegate._parse_userauth_request(m)
    __handler_table = {MSG_SERVICE_REQUEST: _parse_service_request, MSG_USERAUTH_REQUEST: _parse_userauth_request, MSG_USERAUTH_GSSAPI_TOKEN: _parse_userauth_gssapi_token, MSG_USERAUTH_GSSAPI_MIC: _parse_userauth_gssapi_mic}

    @property
    def _handler_table(self):
        return self.__handler_table