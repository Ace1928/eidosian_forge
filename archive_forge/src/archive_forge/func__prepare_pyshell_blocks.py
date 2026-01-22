import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _prepare_pyshell_blocks(self, text):
    """Ensure that Python interactive shell sessions are put in
        code blocks -- even if not properly indented.
        """
    if '>>>' not in text:
        return text
    less_than_tab = self.tab_width - 1
    _pyshell_block_re = re.compile('\n            ^([ ]{0,%d})>>>[ ].*\\n  # first line\n            ^(\\1[^\\S\\n]*\\S.*\\n)*    # any number of subsequent lines with at least one character\n            (?=^\\1?\\n|\\Z)           # ends with a blank line or end of document\n            ' % less_than_tab, re.M | re.X)
    return _pyshell_block_re.sub(self._pyshell_block_sub, text)