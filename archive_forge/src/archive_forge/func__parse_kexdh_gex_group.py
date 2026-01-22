import os
from hashlib import sha1, sha256
from paramiko import util
from paramiko.common import DEBUG, byte_chr, byte_ord, byte_mask
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
def _parse_kexdh_gex_group(self, m):
    self.p = m.get_mpint()
    self.g = m.get_mpint()
    bitlen = util.bit_length(self.p)
    if bitlen < 1024 or bitlen > 8192:
        raise SSHException("Server-generated gex p (don't ask) is out of range ({} bits)".format(bitlen))
    self.transport._log(DEBUG, 'Got server p ({} bits)'.format(bitlen))
    self._generate_x()
    self.e = pow(self.g, self.x, self.p)
    m = Message()
    m.add_byte(c_MSG_KEXDH_GEX_INIT)
    m.add_mpint(self.e)
    self.transport._send_message(m)
    self.transport._expect_packet(_MSG_KEXDH_GEX_REPLY)