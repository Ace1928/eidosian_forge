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
def _ensure_authed(self, ptype, message):
    """
        Checks message type against current auth state.

        If server mode, and auth has not succeeded, and the message is of a
        post-auth type (channel open or global request) an appropriate error
        response Message is crafted and returned to caller for sending.

        Otherwise (client mode, authed, or pre-auth message) returns None.
        """
    if not self.server_mode or ptype <= HIGHEST_USERAUTH_MESSAGE_ID or self.is_authenticated():
        return None
    reply = Message()
    if ptype == MSG_GLOBAL_REQUEST:
        reply.add_byte(cMSG_REQUEST_FAILURE)
    elif ptype == MSG_CHANNEL_OPEN:
        kind = message.get_text()
        chanid = message.get_int()
        reply.add_byte(cMSG_CHANNEL_OPEN_FAILURE)
        reply.add_int(chanid)
        reply.add_int(OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED)
        reply.add_string('')
        reply.add_string('en')
    return reply