import os
from hashlib import sha1
from paramiko import util
from paramiko.common import max_byte, zero_byte, byte_chr, byte_mask
from paramiko.message import Message
from paramiko.ssh_exception import SSHException

Standard SSH key exchange ("kex" if you wanna sound cool).  Diffie-Hellman of
1024 bit key halves, using a known "p" prime and "g" generator.
