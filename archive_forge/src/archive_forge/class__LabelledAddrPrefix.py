import abc
import base64
import collections
import copy
import functools
import io
import itertools
import math
import operator
import re
import socket
import struct
import netaddr
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib.packet import afi as addr_family
from os_ken.lib.packet import safi as subaddr_family
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import stream_parser
from os_ken.lib.packet import vxlan
from os_ken.lib.packet import mpls
from os_ken.lib import addrconv
from os_ken.lib import type_desc
from os_ken.lib.type_desc import TypeDisp
from os_ken.lib import ip
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.utils import binary_str
from os_ken.utils import import_module
class _LabelledAddrPrefix(_AddrPrefix):
    _LABEL_PACK_STR = '!3B'
    _WITHDRAW_LABELS = [8388608, 0]

    def __init__(self, length, addr, labels=None, **kwargs):
        labels = labels if labels else []
        assert isinstance(labels, list)
        is_tuple = isinstance(addr, tuple)
        if is_tuple:
            assert not labels
            labels = addr[0]
            addr = addr[1:]
        else:
            length += struct.calcsize(self._LABEL_PACK_STR) * 8 * len(labels)
        assert length > struct.calcsize(self._LABEL_PACK_STR) * 8 * len(labels)
        prefixes = (labels,)
        super(_LabelledAddrPrefix, self).__init__(prefixes=prefixes, length=length, addr=addr, **kwargs)

    @classmethod
    def _label_to_bin(cls, label):
        buf = bytearray()
        msg_pack_into(cls._LABEL_PACK_STR, buf, 0, (label & 16711680) >> 16, (label & 65280) >> 8, (label & 255) >> 0)
        return bytes(buf)

    @classmethod
    def _label_from_bin(cls, label):
        b1, b2, b3 = struct.unpack_from(cls._LABEL_PACK_STR, bytes(label))
        rest = label[struct.calcsize(cls._LABEL_PACK_STR):]
        return (b1 << 16 | b2 << 8 | b3, rest)

    @classmethod
    def _to_bin(cls, addr):
        labels = addr[0]
        rest = addr[1:]
        labels = [x << 4 for x in labels]
        if labels and labels[-1] not in cls._WITHDRAW_LABELS:
            labels[-1] |= 1
        bin_labels = list((cls._label_to_bin(l) for l in labels))
        return bytes(functools.reduce(lambda x, y: x + y, bin_labels, bytearray()) + cls._prefix_to_bin(rest))

    @classmethod
    def _has_no_label(cls, bin_):
        try:
            length = len(bin_)
            labels = []
            while True:
                label, bin_ = cls._label_from_bin(bin_)
                labels.append(label)
                if label & 1 or label in cls._WITHDRAW_LABELS:
                    break
            assert length > struct.calcsize(cls._LABEL_PACK_STR) * len(labels)
        except struct.error:
            return True
        except AssertionError:
            return True
        return False

    @classmethod
    def _from_bin(cls, addr):
        rest = addr
        labels = []
        if cls._has_no_label(rest):
            return ([],) + cls._prefix_from_bin(rest)
        while True:
            label, rest = cls._label_from_bin(rest)
            labels.append(label >> 4)
            if label & 1 or label in cls._WITHDRAW_LABELS:
                break
        return (labels,) + cls._prefix_from_bin(rest)