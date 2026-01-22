import struct
from io import BytesIO
from paramiko import util
from paramiko.common import zero_byte, max_byte, one_byte
from paramiko.util import u
def add_int(self, n):
    """
        Add an integer to the stream.

        :param int n: integer to add
        """
    self.packet.write(struct.pack('>I', n))
    return self