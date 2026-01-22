from __future__ import annotations
import logging
import os
import shutil
import sys
import tempfile
from email.message import Message
from enum import IntEnum
from io import BytesIO
from numbers import Number
from typing import TYPE_CHECKING
from .decoders import Base64Decoder, QuotedPrintableDecoder
from .exceptions import FileError, FormParserError, MultipartParseError, QuerystringParseError
class QuerystringParser(BaseParser):
    """This is a streaming querystring parser.  It will consume data, and call
    the callbacks given when it has data.

    .. list-table::
       :widths: 15 10 30
       :header-rows: 1

       * - Callback Name
         - Parameters
         - Description
       * - on_field_start
         - None
         - Called when a new field is encountered.
       * - on_field_name
         - data, start, end
         - Called when a portion of a field's name is encountered.
       * - on_field_data
         - data, start, end
         - Called when a portion of a field's data is encountered.
       * - on_field_end
         - None
         - Called when the end of a field is encountered.
       * - on_end
         - None
         - Called when the parser is finished parsing all data.

    :param callbacks: A dictionary of callbacks.  See the documentation for
                      :class:`BaseParser`.

    :param strict_parsing: Whether or not to parse the body strictly.  Defaults
                           to False.  If this is set to True, then the behavior
                           of the parser changes as the following: if a field
                           has a value with an equal sign (e.g. "foo=bar", or
                           "foo="), it is always included.  If a field has no
                           equals sign (e.g. "...&name&..."), it will be
                           treated as an error if 'strict_parsing' is True,
                           otherwise included.  If an error is encountered,
                           then a
                           :class:`multipart.exceptions.QuerystringParseError`
                           will be raised.

    :param max_size: The maximum size of body to parse.  Defaults to infinity -
                     i.e. unbounded.
    """
    state: QuerystringState

    def __init__(self, callbacks: QuerystringCallbacks={}, strict_parsing: bool=False, max_size=float('inf')):
        super().__init__()
        self.state = QuerystringState.BEFORE_FIELD
        self._found_sep = False
        self.callbacks = callbacks
        if not isinstance(max_size, Number) or max_size < 1:
            raise ValueError('max_size must be a positive number, not %r' % max_size)
        self.max_size = max_size
        self._current_size = 0
        self.strict_parsing = strict_parsing

    def write(self, data: bytes) -> int:
        """Write some data to the parser, which will perform size verification,
        parse into either a field name or value, and then pass the
        corresponding data to the underlying callback.  If an error is
        encountered while parsing, a QuerystringParseError will be raised.  The
        "offset" attribute of the raised exception will be set to the offset in
        the input data chunk (NOT the overall stream) that caused the error.

        :param data: a bytestring
        """
        data_len = len(data)
        if self._current_size + data_len > self.max_size:
            new_size = int(self.max_size - self._current_size)
            self.logger.warning('Current size is %d (max %d), so truncating data length from %d to %d', self._current_size, self.max_size, data_len, new_size)
            data_len = new_size
        l = 0
        try:
            l = self._internal_write(data, data_len)
        finally:
            self._current_size += l
        return l

    def _internal_write(self, data: bytes, length: int) -> int:
        state = self.state
        strict_parsing = self.strict_parsing
        found_sep = self._found_sep
        i = 0
        while i < length:
            ch = data[i]
            if state == QuerystringState.BEFORE_FIELD:
                if ch == AMPERSAND or ch == SEMICOLON:
                    if found_sep:
                        if strict_parsing:
                            e = QuerystringParseError('Skipping duplicate ampersand/semicolon at %d' % i)
                            e.offset = i
                            raise e
                        else:
                            self.logger.debug('Skipping duplicate ampersand/semicolon at %d', i)
                    else:
                        found_sep = True
                else:
                    self.callback('field_start')
                    i -= 1
                    state = QuerystringState.FIELD_NAME
                    found_sep = False
            elif state == QuerystringState.FIELD_NAME:
                sep_pos = data.find(b'&', i)
                if sep_pos == -1:
                    sep_pos = data.find(b';', i)
                if sep_pos != -1:
                    equals_pos = data.find(b'=', i, sep_pos)
                else:
                    equals_pos = data.find(b'=', i)
                if equals_pos != -1:
                    self.callback('field_name', data, i, equals_pos)
                    i = equals_pos
                    state = QuerystringState.FIELD_DATA
                elif not strict_parsing:
                    if sep_pos != -1:
                        self.callback('field_name', data, i, sep_pos)
                        self.callback('field_end')
                        i = sep_pos - 1
                        state = QuerystringState.BEFORE_FIELD
                    else:
                        self.callback('field_name', data, i, length)
                        i = length
                else:
                    if sep_pos != -1:
                        e = QuerystringParseError('When strict_parsing is True, we require an equals sign in all field chunks. Did not find one in the chunk that starts at %d' % (i,))
                        e.offset = i
                        raise e
                    self.callback('field_name', data, i, length)
                    i = length
            elif state == QuerystringState.FIELD_DATA:
                sep_pos = data.find(b'&', i)
                if sep_pos == -1:
                    sep_pos = data.find(b';', i)
                if sep_pos != -1:
                    self.callback('field_data', data, i, sep_pos)
                    self.callback('field_end')
                    i = sep_pos - 1
                    state = QuerystringState.BEFORE_FIELD
                else:
                    self.callback('field_data', data, i, length)
                    i = length
            else:
                msg = 'Reached an unknown state %d at %d' % (state, i)
                self.logger.warning(msg)
                e = QuerystringParseError(msg)
                e.offset = i
                raise e
            i += 1
        self.state = state
        self._found_sep = found_sep
        return len(data)

    def finalize(self) -> None:
        """Finalize this parser, which signals to that we are finished parsing,
        if we're still in the middle of a field, an on_field_end callback, and
        then the on_end callback.
        """
        if self.state == QuerystringState.FIELD_DATA:
            self.callback('field_end')
        self.callback('end')

    def __repr__(self) -> str:
        return '{}(strict_parsing={!r}, max_size={!r})'.format(self.__class__.__name__, self.strict_parsing, self.max_size)