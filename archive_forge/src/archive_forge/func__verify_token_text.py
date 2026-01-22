import re
import sys
from weakref import ReferenceType
import weakref
from debian._util import resolve_ref, _strI
from debian._deb822_repro._util import BufferingIterator
def _verify_token_text(self):
    if '\n' in self._text:
        is_single_line_token = False
        if self.is_comment or isinstance(self, Deb822ErrorToken):
            is_single_line_token = True
        if not is_single_line_token and (not self.is_whitespace):
            raise ValueError('Only whitespace, error and comment tokens may contain newlines')
        if not self.text.endswith('\n'):
            raise ValueError('Tokens containing whitespace must end on a newline')
        if is_single_line_token and '\n' in self.text[:-1]:
            raise ValueError('Comments and error tokens must not contain embedded newlines (only end on one)')