import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _encode_backslash_escapes(self, text):
    for ch, escape in list(self._escape_table.items()):
        text = text.replace('\\' + ch, escape)
    return text