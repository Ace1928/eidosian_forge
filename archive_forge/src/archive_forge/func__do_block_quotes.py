import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _do_block_quotes(self, text):
    if '>' not in text:
        return text
    if 'spoiler' in self.extras:
        return self._block_quote_re_spoiler.sub(self._block_quote_sub, text)
    else:
        return self._block_quote_re.sub(self._block_quote_sub, text)