import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _do_code_blocks(self, text):
    """Process Markdown `<pre><code>` blocks."""
    code_block_re = re.compile("\n            (?:\\n\\n|\\A\\n?)\n            (               # $1 = the code block -- one or more lines, starting with a space/tab\n              (?:\n                (?:[ ]{%d} | \\t)  # Lines must start with a tab or a tab-width of spaces\n                .*\\n+\n              )+\n            )\n            ((?=^[ ]{0,%d}\\S)|\\Z)   # Lookahead for non-space at line-start, or end of doc\n            # Lookahead to make sure this block isn't already in a code block.\n            # Needed when syntax highlighting is being used.\n            (?!([^<]|<(/?)span)*\\</code\\>)\n            " % (self.tab_width, self.tab_width), re.M | re.X)
    return code_block_re.sub(self._code_block_sub, text)