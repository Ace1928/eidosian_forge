import struct
from . import packet_base
from os_ken.lib import type_desc

    Converts integer label to binary representation.

    :param mpls_label: MPLS Label.
    :param is_bos: BoS bit.
    :return: Binary representation of label.
    