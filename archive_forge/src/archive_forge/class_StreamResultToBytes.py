import builtins
import codecs
import datetime
import select
import struct
import sys
import zlib
import subunit
import iso8601
class StreamResultToBytes(object):
    """Convert StreamResult API calls to bytes.

    The StreamResult API is defined by testtools.StreamResult.
    """
    status_mask = {None: 0, 'exists': 1, 'inprogress': 2, 'success': 3, 'uxsuccess': 4, 'skip': 5, 'fail': 6, 'xfail': 7}
    zero_b = b'\x00'[0]

    def __init__(self, output_stream):
        """Create a StreamResultToBytes with output written to output_stream.

        :param output_stream: A file-like object. Must support write(bytes)
            and flush() methods. Flush will be called after each write.
            The stream will be passed through subunit.make_stream_binary,
            to handle regular cases such as stdout.
        """
        self.output_stream = subunit.make_stream_binary(output_stream)

    def startTestRun(self):
        pass

    def stopTestRun(self):
        pass

    def status(self, test_id=None, test_status=None, test_tags=None, runnable=True, file_name=None, file_bytes=None, eof=False, mime_type=None, route_code=None, timestamp=None):
        self._write_packet(test_id=test_id, test_status=test_status, test_tags=test_tags, runnable=runnable, file_name=file_name, file_bytes=file_bytes, eof=eof, mime_type=mime_type, route_code=route_code, timestamp=timestamp)

    def _write_utf8(self, a_string, packet):
        utf8 = a_string.encode('utf-8')
        self._write_number(len(utf8), packet)
        packet.append(utf8)

    def _write_len16(self, length, packet):
        assert length < 65536
        packet.append(struct.pack(FMT_16, length))

    def _write_number(self, value, packet):
        packet.extend(self._encode_number(value))

    def _encode_number(self, value):
        assert value >= 0
        if value < 64:
            return [struct.pack(FMT_8, value)]
        elif value < 16384:
            value = value | 16384
            return [struct.pack(FMT_16, value)]
        elif value < 4194304:
            value = value | 8388608
            return [struct.pack(FMT_16, value >> 8), struct.pack(FMT_8, value & 255)]
        elif value < 1073741824:
            value = value | 3221225472
            return [struct.pack(FMT_32, value)]
        else:
            raise ValueError('value too large to encode: %r' % (value,))

    def _write_packet(self, test_id=None, test_status=None, test_tags=None, runnable=True, file_name=None, file_bytes=None, eof=False, mime_type=None, route_code=None, timestamp=None):
        packet = [SIGNATURE]
        packet.append(b'FF')
        packet.append(b'')
        flags = 8192
        if timestamp is not None:
            flags = flags | FLAG_TIMESTAMP
            since_epoch = timestamp - EPOCH
            nanoseconds = since_epoch.microseconds * 1000
            seconds = since_epoch.seconds + since_epoch.days * 24 * 3600
            packet.append(struct.pack(FMT_32, seconds))
            self._write_number(nanoseconds, packet)
        if test_id is not None:
            flags = flags | FLAG_TEST_ID
            self._write_utf8(test_id, packet)
        if test_tags:
            flags = flags | FLAG_TAGS
            self._write_number(len(test_tags), packet)
            for tag in test_tags:
                self._write_utf8(tag, packet)
        if runnable:
            flags = flags | FLAG_RUNNABLE
        if mime_type:
            flags = flags | FLAG_MIME_TYPE
            self._write_utf8(mime_type, packet)
        if file_name is not None:
            flags = flags | FLAG_FILE_CONTENT
            self._write_utf8(file_name, packet)
            self._write_number(len(file_bytes), packet)
            packet.append(file_bytes)
        if eof:
            flags = flags | FLAG_EOF
        if route_code is not None:
            flags = flags | FLAG_ROUTE_CODE
            self._write_utf8(route_code, packet)
        flags = flags | self.status_mask[test_status]
        packet[1] = struct.pack(FMT_16, flags)
        base_length = sum(map(len, packet)) + 4
        if base_length <= 62:
            length_length = 1
        elif base_length <= 16381:
            length_length = 2
        elif base_length <= 4194300:
            length_length = 3
        else:
            raise ValueError('Length too long: %r' % base_length)
        packet[2:3] = self._encode_number(base_length + length_length)
        content = b''.join(packet)
        data = content + struct.pack(FMT_32, zlib.crc32(content) & 4294967295)
        view = memoryview(data)
        datalen = len(data)
        offset = 0
        while offset < datalen:
            written = self.output_stream.write(view[offset:])
            if written is None:
                break
            offset += written
        self.output_stream.flush()