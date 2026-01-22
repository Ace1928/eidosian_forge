from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import gcloud_crcmod as crcmod
import six
def Crc32c(data):
    """Calculates the CRC32C checksum of the provided data.

  Args:
    data: the bytes over which the checksum should be calculated.

  Returns:
    An int representing the CRC32C checksum of the provided bytes.
  """
    crc32c_fun = crcmod.predefined.mkPredefinedCrcFun('crc-32c')
    return crc32c_fun(six.ensure_binary(data))