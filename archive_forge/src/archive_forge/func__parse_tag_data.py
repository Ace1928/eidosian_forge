import datetime
import struct
import sys
from os.path import basename
from typing import List
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
def _parse_tag_data(elem_code, elem_num, raw_data):
    """Return single data value (PRIVATE).

    Arguments:
     - elem_code - What kind of data
     - elem_num - How many data points
     - raw_data - abi file object from which the tags would be unpacked

    """
    if elem_code in _BYTEFMT:
        if elem_num == 1:
            num = ''
        else:
            num = str(elem_num)
        fmt = '>' + num + _BYTEFMT[elem_code]
        assert len(raw_data) == struct.calcsize(fmt)
        data = struct.unpack(fmt, raw_data)
        if elem_code not in [10, 11] and len(data) == 1:
            data = data[0]
        if elem_code == 2:
            return data
        elif elem_code == 10:
            return str(datetime.date(*data))
        elif elem_code == 11:
            return str(datetime.time(*data[:3]))
        elif elem_code == 13:
            return bool(data)
        elif elem_code == 18:
            return data[1:]
        elif elem_code == 19:
            return data[:-1]
        else:
            return data
    else:
        return None