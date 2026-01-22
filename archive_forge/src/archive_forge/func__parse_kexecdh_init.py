import binascii
import hashlib
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import constant_time, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
from paramiko.message import Message
from paramiko.common import byte_chr
from paramiko.ssh_exception import SSHException
def _parse_kexecdh_init(self, m):
    peer_key_bytes = m.get_string()
    peer_key = X25519PublicKey.from_public_bytes(peer_key_bytes)
    K = self._perform_exchange(peer_key)
    K = int(binascii.hexlify(K), 16)
    hm = Message()
    hm.add(self.transport.remote_version, self.transport.local_version, self.transport.remote_kex_init, self.transport.local_kex_init)
    server_key_bytes = self.transport.get_server_key().asbytes()
    exchange_key_bytes = self.key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    hm.add_string(server_key_bytes)
    hm.add_string(peer_key_bytes)
    hm.add_string(exchange_key_bytes)
    hm.add_mpint(K)
    H = self.hash_algo(hm.asbytes()).digest()
    self.transport._set_K_H(K, H)
    sig = self.transport.get_server_key().sign_ssh_data(H, self.transport.host_key_type)
    m = Message()
    m.add_byte(c_MSG_KEXECDH_REPLY)
    m.add_string(server_key_bytes)
    m.add_string(exchange_key_bytes)
    m.add_string(sig)
    self.transport._send_message(m)
    self.transport._activate_outbound()