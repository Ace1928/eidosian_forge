import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _extract_footnote_def_sub(self, match):
    id, text = match.groups()
    text = _dedent(text, skip_first_line=not text.startswith('\n')).strip()
    normed_id = re.sub('\\W', '-', id)
    self.footnotes[normed_id] = text + '\n\n'
    return ''