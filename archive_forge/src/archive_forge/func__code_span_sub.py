import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _code_span_sub(self, match):
    c = match.group(2).strip(' \t')
    c = self._encode_code(c)
    return '<code%s>%s</code>' % (self._html_class_str_from_tag('code'), c)