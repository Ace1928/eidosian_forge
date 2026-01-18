import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def unhash_code(codeblock):
    for key, sanitized in list(self.html_spans.items()):
        codeblock = codeblock.replace(key, sanitized)
    replacements = [('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>')]
    for old, new in replacements:
        codeblock = codeblock.replace(old, new)
    return codeblock