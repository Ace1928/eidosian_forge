import datetime
import struct
import sys
from os.path import basename
from typing import List
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
def _abi_parse_header(header, handle):
    """Return directory contents (PRIVATE)."""
    head_elem_size = header[4]
    head_elem_num = header[5]
    head_offset = header[7]
    index = 0
    while index < head_elem_num:
        start = head_offset + index * head_elem_size
        handle.seek(start)
        dir_entry = struct.unpack(_DIRFMT, handle.read(struct.calcsize(_DIRFMT))) + (start,)
        index += 1
        key = dir_entry[0].decode()
        key += str(dir_entry[1])
        tag_name = dir_entry[0].decode()
        tag_number = dir_entry[1]
        elem_code = dir_entry[2]
        elem_num = dir_entry[4]
        data_size = dir_entry[5]
        data_offset = dir_entry[6]
        tag_offset = dir_entry[8]
        if data_size <= 4:
            data_offset = tag_offset + 20
        handle.seek(data_offset)
        data = handle.read(data_size)
        yield (tag_name, tag_number, _parse_tag_data(elem_code, elem_num, data))