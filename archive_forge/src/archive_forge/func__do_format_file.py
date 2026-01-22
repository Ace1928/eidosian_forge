import argparse
import collections
import contextlib
import io
import re
import tokenize
from typing import TextIO, Tuple
import untokenize  # type: ignore
import docformatter.encode as _encode
import docformatter.strings as _strings
import docformatter.syntax as _syntax
import docformatter.util as _util
def _do_format_file(self, filename):
    """Run format_code() on a file.

        Parameters
        ----------
        filename: str
            The path to the file to be formatted.

        Return
        ------
        result_code: int
            One of the FormatResult codes.
        """
    self.encodor.do_detect_encoding(filename)
    with self.encodor.do_open_with_encoding(filename) as input_file:
        source = input_file.read()
        formatted_source = self._do_format_code(source)
    ret = FormatResult.ok
    show_diff = self.args.diff
    if source != formatted_source:
        ret = FormatResult.check_failed
        if self.args.check:
            print(unicode(filename), file=self.stderror)
        elif self.args.in_place:
            with self.encodor.do_open_with_encoding(filename, mode='w') as output_file:
                output_file.write(formatted_source)
        else:
            show_diff = True
        if show_diff:
            import difflib
            diff = difflib.unified_diff(source.splitlines(), formatted_source.splitlines(), f'before/{filename}', f'after/{filename}', lineterm='')
            self.stdout.write('\n'.join(list(diff) + ['']))
    return ret