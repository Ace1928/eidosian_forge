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
def _parse_channel_open_failure(self, m):
    chanid = m.get_int()
    reason = m.get_int()
    reason_str = m.get_text()
    m.get_text()
    reason_text = CONNECTION_FAILED_CODE.get(reason, '(unknown code)')
    self._log(ERROR, 'Secsh channel {:d} open FAILED: {}: {}'.format(chanid, reason_str, reason_text))
    self.lock.acquire()
    try:
        self.saved_exception = ChannelException(reason, reason_text)
        if chanid in self.channel_events:
            self._channels.delete(chanid)
            if chanid in self.channel_events:
                self.channel_events[chanid].set()
                del self.channel_events[chanid]
    finally:
        self.lock.release()
    return