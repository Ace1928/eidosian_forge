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
def _activate_outbound(self):
    """switch on newly negotiated encryption parameters for
        outbound traffic"""
    m = Message()
    m.add_byte(cMSG_NEWKEYS)
    self._send_message(m)
    if self.agreed_on_strict_kex:
        self._log(DEBUG, 'Resetting outbound seqno after NEWKEYS due to strict mode')
        self.packetizer.reset_seqno_out()
    block_size = self._cipher_info[self.local_cipher]['block-size']
    if self.server_mode:
        IV_out = self._compute_key('B', block_size)
        key_out = self._compute_key('D', self._cipher_info[self.local_cipher]['key-size'])
    else:
        IV_out = self._compute_key('A', block_size)
        key_out = self._compute_key('C', self._cipher_info[self.local_cipher]['key-size'])
    engine = self._get_cipher(self.local_cipher, key_out, IV_out, self._ENCRYPT)
    etm = 'etm@openssh.com' in self.local_mac
    mac_size = self._mac_info[self.local_mac]['size']
    mac_engine = self._mac_info[self.local_mac]['class']
    if self.server_mode:
        mac_key = self._compute_key('F', mac_engine().digest_size)
    else:
        mac_key = self._compute_key('E', mac_engine().digest_size)
    sdctr = self.local_cipher.endswith('-ctr')
    self.packetizer.set_outbound_cipher(engine, block_size, mac_engine, mac_size, mac_key, sdctr, etm=etm)
    compress_out = self._compression_info[self.local_compression][0]
    if compress_out is not None and (self.local_compression != 'zlib@openssh.com' or self.authenticated):
        self._log(DEBUG, 'Switching on outbound compression ...')
        self.packetizer.set_outbound_compressor(compress_out())
    if not self.packetizer.need_rekey():
        self.in_kex = False
    if self.server_mode and self.server_sig_algs and (self._remote_ext_info == 'ext-info-c'):
        extensions = {'server-sig-algs': ','.join(self.preferred_pubkeys)}
        m = Message()
        m.add_byte(cMSG_EXT_INFO)
        m.add_int(len(extensions))
        for name, value in sorted(extensions.items()):
            m.add_string(name)
            m.add_string(value)
        self._send_message(m)
    self._expect_packet(MSG_NEWKEYS)