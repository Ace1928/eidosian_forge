from __future__ import division
import os
import struct
from pyu2f import errors
from pyu2f.hid import base
def ParseReportDescriptor(rd, desc):
    """Parse the binary report descriptor.

  Parse the binary report descriptor into a DeviceDescriptor object.

  Args:
    rd: The binary report descriptor
    desc: The DeviceDescriptor object to update with the results
        from parsing the descriptor.

  Returns:
    None
  """
    rd = bytearray(rd)
    pos = 0
    report_count = None
    report_size = None
    usage_page = None
    usage = None
    while pos < len(rd):
        key = rd[pos]
        key_size, value_length = GetValueLength(rd, pos)
        if key & REPORT_DESCRIPTOR_KEY_MASK == INPUT_ITEM:
            if report_count and report_size:
                byte_length = report_count * report_size // 8
                desc.internal_max_in_report_len = max(desc.internal_max_in_report_len, byte_length)
                report_count = None
                report_size = None
        elif key & REPORT_DESCRIPTOR_KEY_MASK == OUTPUT_ITEM:
            if report_count and report_size:
                byte_length = report_count * report_size // 8
                desc.internal_max_out_report_len = max(desc.internal_max_out_report_len, byte_length)
                report_count = None
                report_size = None
        elif key & REPORT_DESCRIPTOR_KEY_MASK == COLLECTION_ITEM:
            if usage_page:
                desc.usage_page = usage_page
            if usage:
                desc.usage = usage
        elif key & REPORT_DESCRIPTOR_KEY_MASK == REPORT_COUNT:
            if len(rd) >= pos + 1 + value_length:
                report_count = ReadLsbBytes(rd, pos + 1, value_length)
        elif key & REPORT_DESCRIPTOR_KEY_MASK == REPORT_SIZE:
            if len(rd) >= pos + 1 + value_length:
                report_size = ReadLsbBytes(rd, pos + 1, value_length)
        elif key & REPORT_DESCRIPTOR_KEY_MASK == USAGE_PAGE:
            if len(rd) >= pos + 1 + value_length:
                usage_page = ReadLsbBytes(rd, pos + 1, value_length)
        elif key & REPORT_DESCRIPTOR_KEY_MASK == USAGE:
            if len(rd) >= pos + 1 + value_length:
                usage = ReadLsbBytes(rd, pos + 1, value_length)
        pos += value_length + key_size
    return desc