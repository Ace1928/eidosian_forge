import os
from hashlib import sha1, sha256
from paramiko import util
from paramiko.common import DEBUG, byte_chr, byte_ord, byte_mask
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
def _parse_kexdh_gex_init(self, m):
    self.e = m.get_mpint()
    if self.e < 1 or self.e > self.p - 1:
        raise SSHException('Client kex "e" is out of range')
    self._generate_x()
    self.f = pow(self.g, self.x, self.p)
    K = pow(self.e, self.x, self.p)
    key = self.transport.get_server_key().asbytes()
    hm = Message()
    hm.add(self.transport.remote_version, self.transport.local_version, self.transport.remote_kex_init, self.transport.local_kex_init, key)
    if not self.old_style:
        hm.add_int(self.min_bits)
    hm.add_int(self.preferred_bits)
    if not self.old_style:
        hm.add_int(self.max_bits)
    hm.add_mpint(self.p)
    hm.add_mpint(self.g)
    hm.add_mpint(self.e)
    hm.add_mpint(self.f)
    hm.add_mpint(K)
    H = self.hash_algo(hm.asbytes()).digest()
    self.transport._set_K_H(K, H)
    sig = self.transport.get_server_key().sign_ssh_data(H, self.transport.host_key_type)
    m = Message()
    m.add_byte(c_MSG_KEXDH_GEX_REPLY)
    m.add_string(key)
    m.add_mpint(self.f)
    m.add_string(sig)
    self.transport._send_message(m)
    self.transport._activate_outbound()