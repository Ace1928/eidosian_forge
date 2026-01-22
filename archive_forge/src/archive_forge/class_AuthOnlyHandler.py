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
class AuthOnlyHandler(AuthHandler):
    """
    AuthHandler, and just auth, no service requests!

    .. versionadded:: 3.2
    """

    @property
    def _client_handler_table(self):
        my_table = super()._client_handler_table.copy()
        del my_table[MSG_SERVICE_ACCEPT]
        return my_table

    def send_auth_request(self, username, method, finish_message=None):
        """
        Submit a userauth request message & wait for response.

        Performs the transport message send call, sets self.auth_event, and
        will lock-n-block as necessary to both send, and wait for response to,
        the USERAUTH_REQUEST.

        Most callers will want to supply a callback to ``finish_message``,
        which accepts a Message ``m`` and may call mutator methods on it to add
        more fields.
        """
        self.auth_method = method
        self.username = username
        m = Message()
        m.add_byte(cMSG_USERAUTH_REQUEST)
        m.add_string(username)
        m.add_string('ssh-connection')
        m.add_string(method)
        finish_message(m)
        with self.transport.lock:
            self.transport._send_message(m)
        self.auth_event = threading.Event()
        return self.wait_for_response(self.auth_event)

    def auth_none(self, username):
        return self.send_auth_request(username, 'none')

    def auth_publickey(self, username, key):
        key_type, bits = self._get_key_type_and_bits(key)
        algorithm = self._finalize_pubkey_algorithm(key_type)
        blob = self._get_session_blob(key, 'ssh-connection', username, algorithm)

        def finish(m):
            m.add_boolean(True)
            m.add_string(algorithm)
            m.add_string(bits)
            m.add_string(key.sign_ssh_data(blob, algorithm))
        return self.send_auth_request(username, 'publickey', finish)

    def auth_password(self, username, password):

        def finish(m):
            m.add_boolean(False)
            m.add_string(b(password))
        return self.send_auth_request(username, 'password', finish)

    def auth_interactive(self, username, handler, submethods=''):
        """
        response_list = handler(title, instructions, prompt_list)
        """
        self.auth_method = 'keyboard_interactive'
        self.interactive_handler = handler

        def finish(m):
            m.add_string('')
            m.add_string(submethods)
        return self.send_auth_request(username, 'keyboard-interactive', finish)

    def _choose_fallback_pubkey_algorithm(self, key_type, my_algos):
        msg = 'Server did not send a server-sig-algs list; defaulting to something in our preferred algorithms list'
        self._log(DEBUG, msg)
        noncert_key_type = key_type.replace('-cert-v01@openssh.com', '')
        if key_type in my_algos or noncert_key_type in my_algos:
            actual = key_type if key_type in my_algos else noncert_key_type
            msg = f'Current key type, {actual!r}, is in our preferred list; using that'
            algo = actual
        else:
            algo = my_algos[0]
            msg = f'{key_type!r} not in our list - trying first list item instead, {algo!r}'
        self._log(DEBUG, msg)
        return algo