from __future__ import annotations
import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING
from babel.core import Locale
from babel.messages.catalog import Catalog, Message
from babel.util import _cmp, wraptext
class PoFileParser:
    """Support class to  read messages from a ``gettext`` PO (portable object) file
    and add them to a `Catalog`

    See `read_po` for simple cases.
    """
    _keywords = ['msgid', 'msgstr', 'msgctxt', 'msgid_plural']

    def __init__(self, catalog: Catalog, ignore_obsolete: bool=False, abort_invalid: bool=False) -> None:
        self.catalog = catalog
        self.ignore_obsolete = ignore_obsolete
        self.counter = 0
        self.offset = 0
        self.abort_invalid = abort_invalid
        self._reset_message_state()

    def _reset_message_state(self) -> None:
        self.messages = []
        self.translations = []
        self.locations = []
        self.flags = []
        self.user_comments = []
        self.auto_comments = []
        self.context = None
        self.obsolete = False
        self.in_msgid = False
        self.in_msgstr = False
        self.in_msgctxt = False

    def _add_message(self) -> None:
        """
        Add a message to the catalog based on the current parser state and
        clear the state ready to process the next message.
        """
        self.translations.sort()
        if len(self.messages) > 1:
            msgid = tuple((m.denormalize() for m in self.messages))
        else:
            msgid = self.messages[0].denormalize()
        if isinstance(msgid, (list, tuple)):
            string = ['' for _ in range(self.catalog.num_plurals)]
            for idx, translation in self.translations:
                if idx >= self.catalog.num_plurals:
                    self._invalid_pofile('', self.offset, 'msg has more translations than num_plurals of catalog')
                    continue
                string[idx] = translation.denormalize()
            string = tuple(string)
        else:
            string = self.translations[0][1].denormalize()
        msgctxt = self.context.denormalize() if self.context else None
        message = Message(msgid, string, list(self.locations), set(self.flags), self.auto_comments, self.user_comments, lineno=self.offset + 1, context=msgctxt)
        if self.obsolete:
            if not self.ignore_obsolete:
                self.catalog.obsolete[msgid] = message
        else:
            self.catalog[msgid] = message
        self.counter += 1
        self._reset_message_state()

    def _finish_current_message(self) -> None:
        if self.messages:
            self._add_message()

    def _process_message_line(self, lineno, line, obsolete=False) -> None:
        if line.startswith('"'):
            self._process_string_continuation_line(line, lineno)
        else:
            self._process_keyword_line(lineno, line, obsolete)

    def _process_keyword_line(self, lineno, line, obsolete=False) -> None:
        for keyword in self._keywords:
            try:
                if line.startswith(keyword) and line[len(keyword)] in [' ', '[']:
                    arg = line[len(keyword):]
                    break
            except IndexError:
                self._invalid_pofile(line, lineno, 'Keyword must be followed by a string')
        else:
            self._invalid_pofile(line, lineno, "Start of line didn't match any expected keyword.")
            return
        if keyword in ['msgid', 'msgctxt']:
            self._finish_current_message()
        self.obsolete = obsolete
        if keyword == 'msgid':
            self.offset = lineno
        if keyword in ['msgid', 'msgid_plural']:
            self.in_msgctxt = False
            self.in_msgid = True
            self.messages.append(_NormalizedString(arg))
        elif keyword == 'msgstr':
            self.in_msgid = False
            self.in_msgstr = True
            if arg.startswith('['):
                idx, msg = arg[1:].split(']', 1)
                self.translations.append([int(idx), _NormalizedString(msg)])
            else:
                self.translations.append([0, _NormalizedString(arg)])
        elif keyword == 'msgctxt':
            self.in_msgctxt = True
            self.context = _NormalizedString(arg)

    def _process_string_continuation_line(self, line, lineno) -> None:
        if self.in_msgid:
            s = self.messages[-1]
        elif self.in_msgstr:
            s = self.translations[-1][1]
        elif self.in_msgctxt:
            s = self.context
        else:
            self._invalid_pofile(line, lineno, 'Got line starting with " but not in msgid, msgstr or msgctxt')
            return
        s.append(line)

    def _process_comment(self, line) -> None:
        self._finish_current_message()
        if line[1:].startswith(':'):
            for location in line[2:].lstrip().split():
                pos = location.rfind(':')
                if pos >= 0:
                    try:
                        lineno = int(location[pos + 1:])
                    except ValueError:
                        continue
                    self.locations.append((location[:pos], lineno))
                else:
                    self.locations.append((location, None))
        elif line[1:].startswith(','):
            for flag in line[2:].lstrip().split(','):
                self.flags.append(flag.strip())
        elif line[1:].startswith('.'):
            comment = line[2:].strip()
            if comment:
                self.auto_comments.append(comment)
        else:
            self.user_comments.append(line[1:].strip())

    def parse(self, fileobj: IO[AnyStr]) -> None:
        """
        Reads from the file-like object `fileobj` and adds any po file
        units found in it to the `Catalog` supplied to the constructor.
        """
        for lineno, line in enumerate(fileobj):
            line = line.strip()
            if not isinstance(line, str):
                line = line.decode(self.catalog.charset)
            if not line:
                continue
            if line.startswith('#'):
                if line[1:].startswith('~'):
                    self._process_message_line(lineno, line[2:].lstrip(), obsolete=True)
                else:
                    self._process_comment(line)
            else:
                self._process_message_line(lineno, line)
        self._finish_current_message()
        if not self.counter and (self.flags or self.user_comments or self.auto_comments):
            self.messages.append(_NormalizedString('""'))
            self.translations.append([0, _NormalizedString('""')])
            self._add_message()

    def _invalid_pofile(self, line, lineno, msg) -> None:
        assert isinstance(line, str)
        if self.abort_invalid:
            raise PoFileError(msg, self.catalog, line, lineno)
        print('WARNING:', msg)
        print(f'WARNING: Problem on line {lineno + 1}: {line!r}')