import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def _check_word(self, start, end):
    if start.has_tag(self.no_spell_check):
        return
    for tag in self.ignored_tags:
        if start.has_tag(tag):
            return
    word = self._buffer.get_text(start, end, False).strip()
    logger.debug('Checking word %s in range %d:%d to %d:%d.', word, start.get_line(), start.get_line_offset(), end.get_line(), end.get_line_offset())
    if not word:
        return
    if len(self._filters[SpellChecker.FILTER_WORD]):
        if self._regexes[SpellChecker.FILTER_WORD].match(word):
            return
    if len(self._filters[SpellChecker.FILTER_LINE]):
        if _IS_GTK3:
            line_start = self._buffer.get_iter_at_line(start.get_line())
        else:
            _success, line_start = self._buffer.get_iter_at_line(start.get_line())
        line_end = end.copy()
        line_end.forward_to_line_end()
        line = self._buffer.get_text(line_start, line_end, False)
        for match in self._regexes[SpellChecker.FILTER_LINE].finditer(line):
            if match.start() <= start.get_line_offset() <= match.end():
                if _IS_GTK3:
                    start = self._buffer.get_iter_at_line_offset(start.get_line(), match.start())
                    end = self._buffer.get_iter_at_line_offset(start.get_line(), match.end())
                else:
                    _success, start = self._buffer.get_iter_at_line_offset(start.get_line(), match.start())
                    _success, end = self._buffer.get_iter_at_line_offset(start.get_line(), match.end())
                self._buffer.remove_tag(self._misspelled, start, end)
                return
    if len(self._filters[SpellChecker.FILTER_TEXT]):
        text_start, text_end = self._buffer.get_bounds()
        text = self._buffer.get_text(text_start, text_end, False)
        for match in self._regexes[SpellChecker.FILTER_TEXT].finditer(text):
            if match.start() <= start.get_offset() <= match.end():
                start = self._buffer.get_iter_at_offset(match.start())
                end = self._buffer.get_iter_at_offset(match.end())
                self._buffer.remove_tag(self._misspelled, start, end)
                return
    if not self._dictionary.check(word):
        self._buffer.apply_tag(self._misspelled, start, end)