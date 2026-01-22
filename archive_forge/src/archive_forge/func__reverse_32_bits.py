from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import warnings
import six
def _reverse_32_bits(crc_checksum):
    return int('{0:032b}'.format(crc_checksum, width=32)[::-1], 2)