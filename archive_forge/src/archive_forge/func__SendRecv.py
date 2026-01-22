import datetime
import json
import os
import socket
from oauth2client import _helpers
from oauth2client import client
def _SendRecv():
    """Communicate with the Developer Shell server socket."""
    port = int(os.getenv(DEVSHELL_ENV, 0))
    if port == 0:
        raise NoDevshellServer()
    sock = socket.socket()
    sock.connect(('localhost', port))
    data = CREDENTIAL_INFO_REQUEST_JSON
    msg = '{0}\n{1}'.format(len(data), data)
    sock.sendall(_helpers._to_bytes(msg, encoding='utf-8'))
    header = sock.recv(6).decode()
    if '\n' not in header:
        raise CommunicationError('saw no newline in the first 6 bytes')
    len_str, json_str = header.split('\n', 1)
    to_read = int(len_str) - len(json_str)
    if to_read > 0:
        json_str += sock.recv(to_read, socket.MSG_WAITALL).decode()
    return CredentialInfoResponse(json_str)