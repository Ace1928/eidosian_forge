import doctest
import errno
import os
import socket
import subprocess
import sys
import threading
import time
from io import BytesIO
from typing import Optional, Type
from testtools.matchers import DocTestMatches
import breezy
from ... import controldir, debug, errors, osutils, tests
from ... import transport as _mod_transport
from ... import urlutils
from ...tests import features, test_server
from ...transport import local, memory, remote, ssh
from ...transport.http import urllib
from .. import bzrdir
from ..remote import UnknownErrorFromSmartServer
from ..smart import client, medium, message, protocol
from ..smart import request as _mod_request
from ..smart import server as _mod_server
from ..smart import vfs
from . import test_smart
class TestChunkedBodyDecoder(tests.TestCase):
    """Tests for ChunkedBodyDecoder.

    This is the body decoder used for protocol version two.
    """

    def test_construct(self):
        decoder = protocol.ChunkedBodyDecoder()
        self.assertFalse(decoder.finished_reading)
        self.assertEqual(8, decoder.next_read_size())
        self.assertEqual(None, decoder.read_next_chunk())
        self.assertEqual(b'', decoder.unused_data)

    def test_empty_content(self):
        """'chunked
END
' is the complete encoding of a zero-length body.
        """
        decoder = protocol.ChunkedBodyDecoder()
        decoder.accept_bytes(b'chunked\n')
        decoder.accept_bytes(b'END\n')
        self.assertTrue(decoder.finished_reading)
        self.assertEqual(None, decoder.read_next_chunk())
        self.assertEqual(b'', decoder.unused_data)

    def test_one_chunk(self):
        """A body in a single chunk is decoded correctly."""
        decoder = protocol.ChunkedBodyDecoder()
        decoder.accept_bytes(b'chunked\n')
        chunk_length = b'f\n'
        chunk_content = b'123456789abcdef'
        finish = b'END\n'
        decoder.accept_bytes(chunk_length + chunk_content + finish)
        self.assertTrue(decoder.finished_reading)
        self.assertEqual(chunk_content, decoder.read_next_chunk())
        self.assertEqual(b'', decoder.unused_data)

    def test_incomplete_chunk(self):
        """When there are less bytes in the chunk than declared by the length,
        then we haven't finished reading yet.
        """
        decoder = protocol.ChunkedBodyDecoder()
        decoder.accept_bytes(b'chunked\n')
        chunk_length = b'8\n'
        three_bytes = b'123'
        decoder.accept_bytes(chunk_length + three_bytes)
        self.assertFalse(decoder.finished_reading)
        self.assertEqual(5 + 4, decoder.next_read_size(), "The next_read_size hint should be the number of missing bytes in this chunk plus 4 (the length of the end-of-body marker: 'END\\n')")
        self.assertEqual(None, decoder.read_next_chunk())

    def test_incomplete_length(self):
        """A chunk length hasn't been read until a newline byte has been read.
        """
        decoder = protocol.ChunkedBodyDecoder()
        decoder.accept_bytes(b'chunked\n')
        decoder.accept_bytes(b'9')
        self.assertEqual(1, decoder.next_read_size(), "The next_read_size hint should be 1, because we don't know the length yet.")
        decoder.accept_bytes(b'\n')
        self.assertEqual(9 + 4, decoder.next_read_size(), "The next_read_size hint should be the length of the chunk plus 4 (the length of the end-of-body marker: 'END\\n')")
        self.assertFalse(decoder.finished_reading)
        self.assertEqual(None, decoder.read_next_chunk())

    def test_two_chunks(self):
        """Content from multiple chunks is concatenated."""
        decoder = protocol.ChunkedBodyDecoder()
        decoder.accept_bytes(b'chunked\n')
        chunk_one = b'3\naaa'
        chunk_two = b'5\nbbbbb'
        finish = b'END\n'
        decoder.accept_bytes(chunk_one + chunk_two + finish)
        self.assertTrue(decoder.finished_reading)
        self.assertEqual(b'aaa', decoder.read_next_chunk())
        self.assertEqual(b'bbbbb', decoder.read_next_chunk())
        self.assertEqual(None, decoder.read_next_chunk())
        self.assertEqual(b'', decoder.unused_data)

    def test_excess_bytes(self):
        """Bytes after the chunked body are reported as unused bytes."""
        decoder = protocol.ChunkedBodyDecoder()
        decoder.accept_bytes(b'chunked\n')
        chunked_body = b'5\naaaaaEND\n'
        excess_bytes = b'excess bytes'
        decoder.accept_bytes(chunked_body + excess_bytes)
        self.assertTrue(decoder.finished_reading)
        self.assertEqual(b'aaaaa', decoder.read_next_chunk())
        self.assertEqual(excess_bytes, decoder.unused_data)
        self.assertEqual(1, decoder.next_read_size(), 'next_read_size hint should be 1 when finished_reading.')

    def test_multidigit_length(self):
        """Lengths in the chunk prefixes can have multiple digits."""
        decoder = protocol.ChunkedBodyDecoder()
        decoder.accept_bytes(b'chunked\n')
        length = 291
        chunk_prefix = hex(length).encode('ascii') + b'\n'
        chunk_bytes = b'z' * length
        finish = b'END\n'
        decoder.accept_bytes(chunk_prefix + chunk_bytes + finish)
        self.assertTrue(decoder.finished_reading)
        self.assertEqual(chunk_bytes, decoder.read_next_chunk())

    def test_byte_at_a_time(self):
        """A complete body fed to the decoder one byte at a time should not
        confuse the decoder.  That is, it should give the same result as if the
        bytes had been received in one batch.

        This test is the same as test_one_chunk apart from the way accept_bytes
        is called.
        """
        decoder = protocol.ChunkedBodyDecoder()
        decoder.accept_bytes(b'chunked\n')
        chunk_length = b'f\n'
        chunk_content = b'123456789abcdef'
        finish = b'END\n'
        combined = chunk_length + chunk_content + finish
        for i in range(len(combined)):
            decoder.accept_bytes(combined[i:i + 1])
        self.assertTrue(decoder.finished_reading)
        self.assertEqual(chunk_content, decoder.read_next_chunk())
        self.assertEqual(b'', decoder.unused_data)

    def test_read_pending_data_resets(self):
        """read_pending_data does not return the same bytes twice."""
        decoder = protocol.ChunkedBodyDecoder()
        decoder.accept_bytes(b'chunked\n')
        chunk_one = b'3\naaa'
        chunk_two = b'3\nbbb'
        finish = b'END\n'
        decoder.accept_bytes(chunk_one)
        self.assertEqual(b'aaa', decoder.read_next_chunk())
        decoder.accept_bytes(chunk_two)
        self.assertEqual(b'bbb', decoder.read_next_chunk())
        self.assertEqual(None, decoder.read_next_chunk())

    def test_decode_error(self):
        decoder = protocol.ChunkedBodyDecoder()
        decoder.accept_bytes(b'chunked\n')
        chunk_one = b'b\nfirst chunk'
        error_signal = b'ERR\n'
        error_chunks = b'5\npart1' + b'5\npart2'
        finish = b'END\n'
        decoder.accept_bytes(chunk_one + error_signal + error_chunks + finish)
        self.assertTrue(decoder.finished_reading)
        self.assertEqual(b'first chunk', decoder.read_next_chunk())
        expected_failure = _mod_request.FailedSmartServerResponse((b'part1', b'part2'))
        self.assertEqual(expected_failure, decoder.read_next_chunk())

    def test_bad_header(self):
        """accept_bytes raises a SmartProtocolError if a chunked body does not
        start with the right header.
        """
        decoder = protocol.ChunkedBodyDecoder()
        self.assertRaises(errors.SmartProtocolError, decoder.accept_bytes, b'bad header\n')