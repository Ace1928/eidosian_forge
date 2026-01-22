import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _emacs_vars_oneliner_sub(self, match):
    if match.group(1).strip() == '-*-' and match.group(4).strip() == '-*-':
        lead_ws = re.findall('^\\s*', match.group(1))[0]
        tail_ws = re.findall('\\s*$', match.group(4))[0]
        return '%s<!-- %s %s %s -->%s' % (lead_ws, '-*-', match.group(2).strip(), '-*-', tail_ws)
    start, end = match.span()
    return match.string[start:end]