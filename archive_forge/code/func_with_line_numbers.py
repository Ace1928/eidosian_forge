from contextlib import contextmanager
import typing
from .core import (
@staticmethod
def with_line_numbers(s: str, start_line: typing.Optional[int]=None, end_line: typing.Optional[int]=None, expand_tabs: bool=True, eol_mark: str='|', mark_spaces: typing.Optional[str]=None, mark_control: typing.Optional[str]=None) -> str:
    """
        Helpful method for debugging a parser - prints a string with line and column numbers.
        (Line and column numbers are 1-based.)

        :param s: tuple(bool, str - string to be printed with line and column numbers
        :param start_line: int - (optional) starting line number in s to print (default=1)
        :param end_line: int - (optional) ending line number in s to print (default=len(s))
        :param expand_tabs: bool - (optional) expand tabs to spaces, to match the pyparsing default
        :param eol_mark: str - (optional) string to mark the end of lines, helps visualize trailing spaces (default="|")
        :param mark_spaces: str - (optional) special character to display in place of spaces
        :param mark_control: str - (optional) convert non-printing control characters to a placeholding
                                 character; valid values:
                                 - "unicode" - replaces control chars with Unicode symbols, such as "␍" and "␊"
                                 - any single character string - replace control characters with given string
                                 - None (default) - string is displayed as-is

        :return: str - input string with leading line numbers and column number headers
        """
    if expand_tabs:
        s = s.expandtabs()
    if mark_control is not None:
        mark_control = typing.cast(str, mark_control)
        if mark_control == 'unicode':
            transtable_map = {c: u for c, u in zip(range(0, 33), range(9216, 9267))}
            transtable_map[127] = 9249
            tbl = str.maketrans(transtable_map)
            eol_mark = ''
        else:
            ord_mark_control = ord(mark_control)
            tbl = str.maketrans({c: ord_mark_control for c in list(range(0, 32)) + [127]})
        s = s.translate(tbl)
    if mark_spaces is not None and mark_spaces != ' ':
        if mark_spaces == 'unicode':
            tbl = str.maketrans({9: 9225, 32: 9251})
            s = s.translate(tbl)
        else:
            s = s.replace(' ', mark_spaces)
    if start_line is None:
        start_line = 1
    if end_line is None:
        end_line = len(s)
    end_line = min(end_line, len(s))
    start_line = min(max(1, start_line), end_line)
    if mark_control != 'unicode':
        s_lines = s.splitlines()[start_line - 1:end_line]
    else:
        s_lines = [line + '␊' for line in s.split('␊')[start_line - 1:end_line]]
    if not s_lines:
        return ''
    lineno_width = len(str(end_line))
    max_line_len = max((len(line) for line in s_lines))
    lead = ' ' * (lineno_width + 1)
    if max_line_len >= 99:
        header0 = lead + ''.join((f'{' ' * 99}{(i + 1) % 100}' for i in range(max(max_line_len // 100, 1)))) + '\n'
    else:
        header0 = ''
    header1 = header0 + lead + ''.join((f'         {(i + 1) % 10}' for i in range(-(-max_line_len // 10)))) + '\n'
    header2 = lead + '1234567890' * -(-max_line_len // 10) + '\n'
    return header1 + header2 + '\n'.join((f'{i:{lineno_width}d}:{line}{eol_mark}' for i, line in enumerate(s_lines, start=start_line))) + '\n'