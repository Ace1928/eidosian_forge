import os
import socket
import sys
import threading
import time
import weakref
from hashlib import md5, sha1, sha256, sha512
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import algorithms, Cipher, modes
import paramiko
from paramiko import util
from paramiko.auth_handler import AuthHandler, AuthOnlyHandler
from paramiko.ssh_gss import GSSAuth
from paramiko.channel import Channel
from paramiko.common import (
from paramiko.compress import ZlibCompressor, ZlibDecompressor
from paramiko.dsskey import DSSKey
from paramiko.ed25519key import Ed25519Key
from paramiko.kex_curve25519 import KexCurve25519
from paramiko.kex_gex import KexGex, KexGexSHA256
from paramiko.kex_group1 import KexGroup1
from paramiko.kex_group14 import KexGroup14, KexGroup14SHA256
from paramiko.kex_group16 import KexGroup16SHA512
from paramiko.kex_ecdh_nist import KexNistp256, KexNistp384, KexNistp521
from paramiko.kex_gss import KexGSSGex, KexGSSGroup1, KexGSSGroup14
from paramiko.message import Message
from paramiko.packet import Packetizer, NeedRekeyException
from paramiko.primes import ModulusPack
from paramiko.rsakey import RSAKey
from paramiko.ecdsakey import ECDSAKey
from paramiko.server import ServerInterface
from paramiko.sftp_client import SFTPClient
from paramiko.ssh_exception import (
from paramiko.util import (
import atexit
def _parse_global_request(self, m):
    kind = m.get_text()
    self._log(DEBUG, 'Received global request "{}"'.format(kind))
    want_reply = m.get_boolean()
    if not self.server_mode:
        self._log(DEBUG, 'Rejecting "{}" global request from server.'.format(kind))
        ok = False
    elif kind == 'tcpip-forward':
        address = m.get_text()
        port = m.get_int()
        ok = self.server_object.check_port_forward_request(address, port)
        if ok:
            ok = (ok,)
    elif kind == 'cancel-tcpip-forward':
        address = m.get_text()
        port = m.get_int()
        self.server_object.cancel_port_forward_request(address, port)
        ok = True
    else:
        ok = self.server_object.check_global_request(kind, m)
    extra = ()
    if type(ok) is tuple:
        extra = ok
        ok = True
    if want_reply:
        msg = Message()
        if ok:
            msg.add_byte(cMSG_REQUEST_SUCCESS)
            msg.add(*extra)
        else:
            msg.add_byte(cMSG_REQUEST_FAILURE)
        self._send_message(msg)