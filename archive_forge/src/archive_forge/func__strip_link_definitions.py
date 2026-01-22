import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _strip_link_definitions(self, text):
    less_than_tab = self.tab_width - 1
    _link_def_re = re.compile('\n            ^[ ]{0,%d}\\[(.+)\\]: # id = \\1\n              [ \\t]*\n              \\n?               # maybe *one* newline\n              [ \\t]*\n            <?(.+?)>?           # url = \\2\n              [ \\t]*\n            (?:\n                \\n?             # maybe one newline\n                [ \\t]*\n                (?<=\\s)         # lookbehind for whitespace\n                [\'"(]\n                ([^\\n]*)        # title = \\3\n                [\'")]\n                [ \\t]*\n            )?  # title is optional\n            (?:\\n+|\\Z)\n            ' % less_than_tab, re.X | re.M | re.U)
    return _link_def_re.sub(self._extract_link_def_sub, text)