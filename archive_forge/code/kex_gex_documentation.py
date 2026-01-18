import os
from hashlib import sha1, sha256
from paramiko import util
from paramiko.common import DEBUG, byte_chr, byte_ord, byte_mask
from paramiko.message import Message
from paramiko.ssh_exception import SSHException

Variant on `KexGroup1 <paramiko.kex_group1.KexGroup1>` where the prime "p" and
generator "g" are provided by the server.  A bit more work is required on the
client side, and a **lot** more on the server side.
