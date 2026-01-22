import codecs
import functools
import os
import pickle
import re
import sys
import textwrap
import zipfile
from abc import ABCMeta, abstractmethod
from gzip import WRITE as GZ_WRITE
from gzip import GzipFile
from io import BytesIO, TextIOWrapper
from urllib.request import url2pathname, urlopen
from nltk import grammar, sem
from nltk.compat import add_py3_data, py3_data
from nltk.internals import deprecated
class SeekableUnicodeStreamReader:
    """
    A stream reader that automatically encodes the source byte stream
    into unicode (like ``codecs.StreamReader``); but still supports the
    ``seek()`` and ``tell()`` operations correctly.  This is in contrast
    to ``codecs.StreamReader``, which provide *broken* ``seek()`` and
    ``tell()`` methods.

    This class was motivated by ``StreamBackedCorpusView``, which
    makes extensive use of ``seek()`` and ``tell()``, and needs to be
    able to handle unicode-encoded files.

    Note: this class requires stateless decoders.  To my knowledge,
    this shouldn't cause a problem with any of python's builtin
    unicode encodings.
    """
    DEBUG = True

    @py3_data
    def __init__(self, stream, encoding, errors='strict'):
        stream.seek(0)
        self.stream = stream
        'The underlying stream.'
        self.encoding = encoding
        'The name of the encoding that should be used to encode the\n           underlying stream.'
        self.errors = errors
        "The error mode that should be used when decoding data from\n           the underlying stream.  Can be 'strict', 'ignore', or\n           'replace'."
        self.decode = codecs.getdecoder(encoding)
        'The function that is used to decode byte strings into\n           unicode strings.'
        self.bytebuffer = b''
        'A buffer to use bytes that have been read but have not yet\n           been decoded.  This is only used when the final bytes from\n           a read do not form a complete encoding for a character.'
        self.linebuffer = None
        'A buffer used by ``readline()`` to hold characters that have\n           been read, but have not yet been returned by ``read()`` or\n           ``readline()``.  This buffer consists of a list of unicode\n           strings, where each string corresponds to a single line.\n           The final element of the list may or may not be a complete\n           line.  Note that the existence of a linebuffer makes the\n           ``tell()`` operation more complex, because it must backtrack\n           to the beginning of the buffer to determine the correct\n           file position in the underlying byte stream.'
        self._rewind_checkpoint = 0
        'The file position at which the most recent read on the\n           underlying stream began.  This is used, together with\n           ``_rewind_numchars``, to backtrack to the beginning of\n           ``linebuffer`` (which is required by ``tell()``).'
        self._rewind_numchars = None
        'The number of characters that have been returned since the\n           read that started at ``_rewind_checkpoint``.  This is used,\n           together with ``_rewind_checkpoint``, to backtrack to the\n           beginning of ``linebuffer`` (which is required by ``tell()``).'
        self._bom = self._check_bom()
        'The length of the byte order marker at the beginning of\n           the stream (or None for no byte order marker).'

    def read(self, size=None):
        """
        Read up to ``size`` bytes, decode them using this reader's
        encoding, and return the resulting unicode string.

        :param size: The maximum number of bytes to read.  If not
            specified, then read as many bytes as possible.
        :type size: int
        :rtype: unicode
        """
        chars = self._read(size)
        if self.linebuffer:
            chars = ''.join(self.linebuffer) + chars
            self.linebuffer = None
            self._rewind_numchars = None
        return chars

    def discard_line(self):
        if self.linebuffer and len(self.linebuffer) > 1:
            line = self.linebuffer.pop(0)
            self._rewind_numchars += len(line)
        else:
            self.stream.readline()

    def readline(self, size=None):
        """
        Read a line of text, decode it using this reader's encoding,
        and return the resulting unicode string.

        :param size: The maximum number of bytes to read.  If no
            newline is encountered before ``size`` bytes have been read,
            then the returned value may not be a complete line of text.
        :type size: int
        """
        if self.linebuffer and len(self.linebuffer) > 1:
            line = self.linebuffer.pop(0)
            self._rewind_numchars += len(line)
            return line
        readsize = size or 72
        chars = ''
        if self.linebuffer:
            chars += self.linebuffer.pop()
            self.linebuffer = None
        while True:
            startpos = self.stream.tell() - len(self.bytebuffer)
            new_chars = self._read(readsize)
            if new_chars and new_chars.endswith('\r'):
                new_chars += self._read(1)
            chars += new_chars
            lines = chars.splitlines(True)
            if len(lines) > 1:
                line = lines[0]
                self.linebuffer = lines[1:]
                self._rewind_numchars = len(new_chars) - (len(chars) - len(line))
                self._rewind_checkpoint = startpos
                break
            elif len(lines) == 1:
                line0withend = lines[0]
                line0withoutend = lines[0].splitlines(False)[0]
                if line0withend != line0withoutend:
                    line = line0withend
                    break
            if not new_chars or size is not None:
                line = chars
                break
            if readsize < 8000:
                readsize *= 2
        return line

    def readlines(self, sizehint=None, keepends=True):
        """
        Read this file's contents, decode them using this reader's
        encoding, and return it as a list of unicode lines.

        :rtype: list(unicode)
        :param sizehint: Ignored.
        :param keepends: If false, then strip newlines.
        """
        return self.read().splitlines(keepends)

    def next(self):
        """Return the next decoded line from the underlying stream."""
        line = self.readline()
        if line:
            return line
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def __iter__(self):
        """Return self"""
        return self

    def __del__(self):
        if not self.closed:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def xreadlines(self):
        """Return self"""
        return self

    @property
    def closed(self):
        """True if the underlying stream is closed."""
        return self.stream.closed

    @property
    def name(self):
        """The name of the underlying stream."""
        return self.stream.name

    @property
    def mode(self):
        """The mode of the underlying stream."""
        return self.stream.mode

    def close(self):
        """
        Close the underlying stream.
        """
        self.stream.close()

    def seek(self, offset, whence=0):
        """
        Move the stream to a new file position.  If the reader is
        maintaining any buffers, then they will be cleared.

        :param offset: A byte count offset.
        :param whence: If 0, then the offset is from the start of the file
            (offset should be positive), if 1, then the offset is from the
            current position (offset may be positive or negative); and if 2,
            then the offset is from the end of the file (offset should
            typically be negative).
        """
        if whence == 1:
            raise ValueError('Relative seek is not supported for SeekableUnicodeStreamReader -- consider using char_seek_forward() instead.')
        self.stream.seek(offset, whence)
        self.linebuffer = None
        self.bytebuffer = b''
        self._rewind_numchars = None
        self._rewind_checkpoint = self.stream.tell()

    def char_seek_forward(self, offset):
        """
        Move the read pointer forward by ``offset`` characters.
        """
        if offset < 0:
            raise ValueError('Negative offsets are not supported')
        self.seek(self.tell())
        self._char_seek_forward(offset)

    def _char_seek_forward(self, offset, est_bytes=None):
        """
        Move the file position forward by ``offset`` characters,
        ignoring all buffers.

        :param est_bytes: A hint, giving an estimate of the number of
            bytes that will be needed to move forward by ``offset`` chars.
            Defaults to ``offset``.
        """
        if est_bytes is None:
            est_bytes = offset
        bytes = b''
        while True:
            newbytes = self.stream.read(est_bytes - len(bytes))
            bytes += newbytes
            chars, bytes_decoded = self._incr_decode(bytes)
            if len(chars) == offset:
                self.stream.seek(-len(bytes) + bytes_decoded, 1)
                return
            if len(chars) > offset:
                while len(chars) > offset:
                    est_bytes += offset - len(chars)
                    chars, bytes_decoded = self._incr_decode(bytes[:est_bytes])
                self.stream.seek(-len(bytes) + bytes_decoded, 1)
                return
            est_bytes += offset - len(chars)

    def tell(self):
        """
        Return the current file position on the underlying byte
        stream.  If this reader is maintaining any buffers, then the
        returned file position will be the position of the beginning
        of those buffers.
        """
        if self.linebuffer is None:
            return self.stream.tell() - len(self.bytebuffer)
        orig_filepos = self.stream.tell()
        bytes_read = orig_filepos - len(self.bytebuffer) - self._rewind_checkpoint
        buf_size = sum((len(line) for line in self.linebuffer))
        est_bytes = int(bytes_read * self._rewind_numchars / (self._rewind_numchars + buf_size))
        self.stream.seek(self._rewind_checkpoint)
        self._char_seek_forward(self._rewind_numchars, est_bytes)
        filepos = self.stream.tell()
        if self.DEBUG:
            self.stream.seek(filepos)
            check1 = self._incr_decode(self.stream.read(50))[0]
            check2 = ''.join(self.linebuffer)
            assert check1.startswith(check2) or check2.startswith(check1)
        self.stream.seek(orig_filepos)
        return filepos

    def _read(self, size=None):
        """
        Read up to ``size`` bytes from the underlying stream, decode
        them using this reader's encoding, and return the resulting
        unicode string.  ``linebuffer`` is not included in the result.
        """
        if size == 0:
            return ''
        if self._bom and self.stream.tell() == 0:
            self.stream.read(self._bom)
        if size is None:
            new_bytes = self.stream.read()
        else:
            new_bytes = self.stream.read(size)
        bytes = self.bytebuffer + new_bytes
        chars, bytes_decoded = self._incr_decode(bytes)
        if size is not None and (not chars) and (len(new_bytes) > 0):
            while not chars:
                new_bytes = self.stream.read(1)
                if not new_bytes:
                    break
                bytes += new_bytes
                chars, bytes_decoded = self._incr_decode(bytes)
        self.bytebuffer = bytes[bytes_decoded:]
        return chars

    def _incr_decode(self, bytes):
        """
        Decode the given byte string into a unicode string, using this
        reader's encoding.  If an exception is encountered that
        appears to be caused by a truncation error, then just decode
        the byte string without the bytes that cause the trunctaion
        error.

        Return a tuple ``(chars, num_consumed)``, where ``chars`` is
        the decoded unicode string, and ``num_consumed`` is the
        number of bytes that were consumed.
        """
        while True:
            try:
                return self.decode(bytes, 'strict')
            except UnicodeDecodeError as exc:
                if exc.end == len(bytes):
                    return self.decode(bytes[:exc.start], self.errors)
                elif self.errors == 'strict':
                    raise
                else:
                    return self.decode(bytes, self.errors)
    _BOM_TABLE = {'utf8': [(codecs.BOM_UTF8, None)], 'utf16': [(codecs.BOM_UTF16_LE, 'utf16-le'), (codecs.BOM_UTF16_BE, 'utf16-be')], 'utf16le': [(codecs.BOM_UTF16_LE, None)], 'utf16be': [(codecs.BOM_UTF16_BE, None)], 'utf32': [(codecs.BOM_UTF32_LE, 'utf32-le'), (codecs.BOM_UTF32_BE, 'utf32-be')], 'utf32le': [(codecs.BOM_UTF32_LE, None)], 'utf32be': [(codecs.BOM_UTF32_BE, None)]}

    def _check_bom(self):
        enc = re.sub('[ -]', '', self.encoding.lower())
        bom_info = self._BOM_TABLE.get(enc)
        if bom_info:
            bytes = self.stream.read(16)
            self.stream.seek(0)
            for bom, new_encoding in bom_info:
                if bytes.startswith(bom):
                    if new_encoding:
                        self.encoding = new_encoding
                    return len(bom)
        return None