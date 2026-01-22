import os
from hashlib import sha1
from paramiko.common import (
from paramiko import util
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
def _parse_kexgss_hostkey(self, m):
    """
        Parse the SSH2_MSG_KEXGSS_HOSTKEY message (client mode).

        :param `Message` m: The content of the SSH2_MSG_KEXGSS_HOSTKEY message
        """
    host_key = m.get_string()
    self.transport.host_key = host_key
    sig = m.get_string()
    self.transport._verify_key(host_key, sig)
    self.transport._expect_packet(MSG_KEXGSS_CONTINUE, MSG_KEXGSS_COMPLETE)