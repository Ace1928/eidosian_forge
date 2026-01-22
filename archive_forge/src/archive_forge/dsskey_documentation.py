from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import dsa
from cryptography.hazmat.primitives.asymmetric.utils import (
from paramiko import util
from paramiko.common import zero_byte
from paramiko.ssh_exception import SSHException
from paramiko.message import Message
from paramiko.ber import BER, BERException
from paramiko.pkey import PKey

        Generate a new private DSS key.  This factory function can be used to
        generate a new host key or authentication key.

        :param int bits: number of bits the generated key should be.
        :param progress_func: Unused
        :return: new `.DSSKey` private key
        