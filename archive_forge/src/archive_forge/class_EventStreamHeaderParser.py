from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
class EventStreamHeaderParser:
    """Parses the event headers from an event stream message.

    Expects all of the header data upfront and creates a dictionary of headers
    to return. This object can be reused multiple times to parse the headers
    from multiple event stream messages.
    """
    _HEADER_TYPE_MAP = {0: DecodeUtils.unpack_true, 1: DecodeUtils.unpack_false, 2: DecodeUtils.unpack_int8, 3: DecodeUtils.unpack_int16, 4: DecodeUtils.unpack_int32, 5: DecodeUtils.unpack_int64, 6: DecodeUtils.unpack_byte_array, 7: DecodeUtils.unpack_utf8_string, 8: DecodeUtils.unpack_int64, 9: DecodeUtils.unpack_uuid}

    def __init__(self):
        self._data = None

    def parse(self, data):
        """Parses the event stream headers from an event stream message.

        :type data: bytes
        :param data: The bytes that correspond to the headers section of an
        event stream message.

        :rtype: dict
        :returns: A dictionary of header key, value pairs.
        """
        self._data = data
        return self._parse_headers()

    def _parse_headers(self):
        headers = {}
        while self._data:
            name, value = self._parse_header()
            if name in headers:
                raise DuplicateHeader(name)
            headers[name] = value
        return headers

    def _parse_header(self):
        name = self._parse_name()
        value = self._parse_value()
        return (name, value)

    def _parse_name(self):
        name, consumed = DecodeUtils.unpack_utf8_string(self._data, 1)
        self._advance_data(consumed)
        return name

    def _parse_type(self):
        type, consumed = DecodeUtils.unpack_uint8(self._data)
        self._advance_data(consumed)
        return type

    def _parse_value(self):
        header_type = self._parse_type()
        value_unpacker = self._HEADER_TYPE_MAP[header_type]
        value, consumed = value_unpacker(self._data)
        self._advance_data(consumed)
        return value

    def _advance_data(self, consumed):
        self._data = self._data[consumed:]