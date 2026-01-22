from __future__ import annotations
from ruamel.yaml.error import MarkedYAMLError, CommentMark  # NOQA
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.docinfo import Version, Tag  # NOQA
from ruamel.yaml.compat import check_anchorname_char, _debug, nprint, nprintf  # NOQA
class RoundTripScannerSC(Scanner):

    def __init__(self, *arg: Any, **kw: Any) -> None:
        super().__init__(*arg, **kw)
        assert self.loader is not None
        self.comments = None

    def get_token(self) -> Any:
        while self.need_more_tokens():
            self.fetch_more_tokens()
        if len(self.tokens) > 0:
            if isinstance(self.tokens[0], BlockEndToken):
                self.comments.assign_post(self.tokens[0])
            else:
                self.comments.assign_pre(self.tokens[0])
            self.tokens_taken += 1
            return self.tokens.pop(0)

    def need_more_tokens(self) -> bool:
        if self.comments is None:
            self.loader.parsed_comments = self.comments = ScannedComments()
        if self.done:
            return False
        if len(self.tokens) == 0:
            return True
        self.stale_possible_simple_keys()
        if self.next_possible_simple_key() == self.tokens_taken:
            return True
        if len(self.tokens) < 2:
            return True
        if self.tokens[0].start_mark.line == self.tokens[-1].start_mark.line:
            return True
        if True:
            if _debug != 0:
                xprintf('-x--', len(self.tokens))
                for t in self.tokens:
                    xprintf(t)
                xprintf(self.comments.str_unprocessed())
        self.comments.assign_pre(self.tokens[0])
        self.comments.assign_eol(self.tokens)
        return False

    def scan_to_next_token(self) -> None:
        srp = self.reader.peek
        srf = self.reader.forward
        if self.reader.index == 0 and srp() == '\ufeff':
            srf()
        start_mark = self.reader.get_mark()
        found = False
        while not found:
            while srp() == ' ':
                srf()
            ch = srp()
            if ch == '#':
                comment_start_mark = self.reader.get_mark()
                comment = ch
                srf()
                while ch not in _THE_END:
                    ch = srp()
                    if ch == '\x00':
                        comment += '\n'
                        break
                    comment += ch
                    srf()
                if start_mark.column == 0:
                    self.comments.add_full_line_comment(comment, comment_start_mark.column, comment_start_mark.line)
                else:
                    self.comments.add_eol_comment(comment, comment_start_mark.column, comment_start_mark.line)
                    comment = ''
                self.scan_empty_or_full_line_comments()
                if not self.flow_level:
                    self.allow_simple_key = True
                return
            if bool(self.scan_line_break()):
                if not self.flow_level:
                    self.allow_simple_key = True
                self.scan_empty_or_full_line_comments()
                return None
                ch = srp()
                if ch == '\n':
                    start_mark = self.reader.get_mark()
                    comment = ''
                    while ch:
                        ch = self.scan_line_break(empty_line=True)
                        comment += ch
                    if srp() == '#':
                        comment = comment.rsplit('\n', 1)[0] + '\n'
                    _ = self.reader.get_mark()
                    return None
            else:
                found = True
        return None

    def scan_empty_or_full_line_comments(self) -> None:
        blmark = self.reader.get_mark()
        assert blmark.column == 0
        blanks = ''
        comment = None
        mark = None
        ch = self.reader.peek()
        while True:
            if ch in '\r\n\x85\u2028\u2029':
                if self.reader.prefix(2) == '\r\n':
                    self.reader.forward(2)
                else:
                    self.reader.forward()
                if comment is not None:
                    comment += '\n'
                    self.comments.add_full_line_comment(comment, mark.column, mark.line)
                    comment = None
                else:
                    blanks += '\n'
                    self.comments.add_blank_line(blanks, blmark.column, blmark.line)
                blanks = ''
                blmark = self.reader.get_mark()
                ch = self.reader.peek()
                continue
            if comment is None:
                if ch in ' \t':
                    blanks += ch
                elif ch == '#':
                    mark = self.reader.get_mark()
                    comment = '#'
                else:
                    break
            else:
                comment += ch
            self.reader.forward()
            ch = self.reader.peek()

    def scan_block_scalar_ignored_line(self, start_mark: Any) -> Any:
        srp = self.reader.peek
        srf = self.reader.forward
        prefix = ''
        comment = None
        while srp() == ' ':
            prefix += srp()
            srf()
        if srp() == '#':
            comment = ''
            mark = self.reader.get_mark()
            while srp() not in _THE_END:
                comment += srp()
                srf()
            comment += '\n'
        ch = srp()
        if ch not in _THE_END:
            raise ScannerError('while scanning a block scalar', start_mark, f'expected a comment or a line break, but found {ch!r}', self.reader.get_mark())
        if comment is not None:
            self.comments.add_eol_comment(comment, mark.column, mark.line)
        self.scan_line_break()
        return None