import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _toc_add_entry(self, level, id, name):
    if level > self._toc_depth:
        return
    if self._toc is None:
        self._toc = []
    self._toc.append((level, id, self._unescape_special_chars(name)))