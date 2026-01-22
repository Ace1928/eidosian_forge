import binascii
import hashlib
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import constant_time, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
from paramiko.message import Message
from paramiko.common import byte_chr
from paramiko.ssh_exception import SSHException
def _parse_kexecdh_reply(self, m):
    peer_host_key_bytes = m.get_string()
    peer_key_bytes = m.get_string()
    sig = m.get_binary()
    peer_key = X25519PublicKey.from_public_bytes(peer_key_bytes)
    K = self._perform_exchange(peer_key)
    K = int(binascii.hexlify(K), 16)
    hm = Message()
    hm.add(self.transport.local_version, self.transport.remote_version, self.transport.local_kex_init, self.transport.remote_kex_init)
    hm.add_string(peer_host_key_bytes)
    hm.add_string(self.key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw))
    hm.add_string(peer_key_bytes)
    hm.add_mpint(K)
    self.transport._set_K_H(K, self.hash_algo(hm.asbytes()).digest())
    self.transport._verify_key(peer_host_key_bytes, sig)
    self.transport._activate_outbound()