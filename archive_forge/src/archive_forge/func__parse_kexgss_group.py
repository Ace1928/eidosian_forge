import os
from hashlib import sha1
from paramiko.common import (
from paramiko import util
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
def _parse_kexgss_group(self, m):
    """
        Parse the SSH2_MSG_KEXGSS_GROUP message (client mode).

        :param `Message` m: The content of the SSH2_MSG_KEXGSS_GROUP message
        """
    self.p = m.get_mpint()
    self.g = m.get_mpint()
    bitlen = util.bit_length(self.p)
    if bitlen < 1024 or bitlen > 8192:
        raise SSHException("Server-generated gex p (don't ask) is out of range ({} bits)".format(bitlen))
    self.transport._log(DEBUG, 'Got server p ({} bits)'.format(bitlen))
    self._generate_x()
    self.e = pow(self.g, self.x, self.p)
    m = Message()
    m.add_byte(c_MSG_KEXGSS_INIT)
    m.add_string(self.kexgss.ssh_init_sec_context(target=self.gss_host))
    m.add_mpint(self.e)
    self.transport._send_message(m)
    self.transport._expect_packet(MSG_KEXGSS_HOSTKEY, MSG_KEXGSS_CONTINUE, MSG_KEXGSS_COMPLETE, MSG_KEXGSS_ERROR)