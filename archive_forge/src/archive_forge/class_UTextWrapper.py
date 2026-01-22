import textwrap
from unicodedata import east_asian_width as _eawidth
from . import osutils
class UTextWrapper(textwrap.TextWrapper):
    """
    Extend TextWrapper for Unicode.

    This textwrapper handles east asian double width and split word
    even if !break_long_words when word contains double width
    characters.

    :param ambiguous_width: (keyword argument) width for character when
                            unicodedata.east_asian_width(c) == 'A'
                            (default: 1)

    Limitations:
    * expand_tabs doesn't fixed. It uses len() for calculating width
      of string on left of TAB.
    * Handles one codeunit as a single character having 1 or 2 width.
      This is not correct when there are surrogate pairs, combined
      characters or zero-width characters.
    * Treats all asian character are line breakable. But it is not
      true because line breaking is prohibited around some characters.
      (For example, breaking before punctation mark is prohibited.)
      See UAX # 14 "UNICODE LINE BREAKING ALGORITHM"
    """

    def __init__(self, width=None, **kwargs):
        if width is None:
            width = (osutils.terminal_width() or osutils.default_terminal_width) - 1
        ambi_width = kwargs.pop('ambiguous_width', 1)
        if ambi_width == 1:
            self._east_asian_doublewidth = 'FW'
        elif ambi_width == 2:
            self._east_asian_doublewidth = 'FWA'
        else:
            raise ValueError('ambiguous_width should be 1 or 2')
        self.max_lines = kwargs.get('max_lines', None)
        textwrap.TextWrapper.__init__(self, width, **kwargs)

    def _unicode_char_width(self, uc):
        """Return width of character `uc`.

        :param:     uc      Single unicode character.
        """
        return _eawidth(uc) in self._east_asian_doublewidth and 2 or 1

    def _width(self, s):
        """Returns width for s.

        When s is unicode, take care of east asian width.
        When s is bytes, treat all byte is single width character.
        """
        charwidth = self._unicode_char_width
        return sum((charwidth(c) for c in s))

    def _cut(self, s, width):
        """Returns head and rest of s. (head+rest == s)

        Head is large as long as _width(head) <= width.
        """
        w = 0
        charwidth = self._unicode_char_width
        for pos, c in enumerate(s):
            w += charwidth(c)
            if w > width:
                return (s[:pos], s[pos:])
        return (s, '')

    def _fix_sentence_endings(self, chunks):
        """_fix_sentence_endings(chunks : [string])

        Correct for sentence endings buried in 'chunks'.  Eg. when the
        original text contains "... foo.
Bar ...", munge_whitespace()
        and split() will convert that to [..., "foo.", " ", "Bar", ...]
        which has one too few spaces; this method simply changes the one
        space to two.

        Note: This function is copied from textwrap.TextWrap and modified
        to use unicode always.
        """
        i = 0
        L = len(chunks) - 1
        patsearch = self.sentence_end_re.search
        while i < L:
            if chunks[i + 1] == ' ' and patsearch(chunks[i]):
                chunks[i + 1] = '  '
                i += 2
            else:
                i += 1

    def _handle_long_word(self, chunks, cur_line, cur_len, width):
        if width < 2:
            space_left = chunks[-1] and self._width(chunks[-1][0]) or 1
        else:
            space_left = width - cur_len
        if self.break_long_words:
            head, rest = self._cut(chunks[-1], space_left)
            cur_line.append(head)
            if rest:
                chunks[-1] = rest
            else:
                del chunks[-1]
        elif not cur_line:
            cur_line.append(chunks.pop())

    def _wrap_chunks(self, chunks):
        lines = []
        if self.width <= 0:
            raise ValueError('invalid width %r (must be > 0)' % self.width)
        if self.max_lines is not None:
            if self.max_lines > 1:
                indent = self.subsequent_indent
            else:
                indent = self.initial_indent
            if self._width(indent) + self._width(self.placeholder.lstrip()) > self.width:
                raise ValueError('placeholder too large for max width')
        chunks.reverse()
        while chunks:
            cur_line = []
            cur_len = 0
            if lines:
                indent = self.subsequent_indent
            else:
                indent = self.initial_indent
            width = self.width - len(indent)
            if self.drop_whitespace and chunks[-1].strip() == '' and lines:
                del chunks[-1]
            while chunks:
                l = self._width(chunks[-1])
                if cur_len + l <= width:
                    cur_line.append(chunks.pop())
                    cur_len += l
                else:
                    break
            if chunks and self._width(chunks[-1]) > width:
                self._handle_long_word(chunks, cur_line, cur_len, width)
                cur_len = sum(map(len, cur_line))
            if self.drop_whitespace and cur_line and (not cur_line[-1].strip()):
                del cur_line[-1]
            if cur_line:
                if self.max_lines is None or len(lines) + 1 < self.max_lines or ((not chunks or (self.drop_whitespace and len(chunks) == 1 and (not chunks[0].strip()))) and cur_len <= width):
                    lines.append(indent + ''.join(cur_line))
                else:
                    while cur_line:
                        if cur_line[-1].strip() and cur_len + self._width(self.placeholder) <= width:
                            cur_line.append(self.placeholder)
                            lines.append(indent + ''.join(cur_line))
                            break
                        cur_len -= self._width(cur_line[-1])
                        del cur_line[-1]
                    else:
                        if lines:
                            prev_line = lines[-1].rstrip()
                            if self._width(prev_line) + self._width(self.placeholder) <= self.width:
                                lines[-1] = prev_line + self.placeholder
                                break
                        lines.append(indent + self.placeholder.lstrip())
                    break
        return lines

    def _split(self, text):
        chunks = textwrap.TextWrapper._split(self, osutils.safe_unicode(text))
        cjk_split_chunks = []
        for chunk in chunks:
            prev_pos = 0
            for pos, char in enumerate(chunk):
                if self._unicode_char_width(char) == 2:
                    if prev_pos < pos:
                        cjk_split_chunks.append(chunk[prev_pos:pos])
                    cjk_split_chunks.append(char)
                    prev_pos = pos + 1
            if prev_pos < len(chunk):
                cjk_split_chunks.append(chunk[prev_pos:])
        return cjk_split_chunks

    def wrap(self, text):
        return textwrap.TextWrapper.wrap(self, osutils.safe_unicode(text))