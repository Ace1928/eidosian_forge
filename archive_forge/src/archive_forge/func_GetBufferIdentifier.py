from . import encode
from . import number_types
from . import packer
def GetBufferIdentifier(buf, offset, size_prefixed=False):
    """Extract the file_identifier from a buffer"""
    if size_prefixed:
        offset += number_types.UOffsetTFlags.bytewidth
    offset += number_types.UOffsetTFlags.bytewidth
    end = offset + encode.FILE_IDENTIFIER_LENGTH
    return buf[offset:end]