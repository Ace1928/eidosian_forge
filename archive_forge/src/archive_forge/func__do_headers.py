import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _do_headers(self, text):
    if 'tag-friendly' in self.extras:
        return self._h_re_tag_friendly.sub(self._h_sub, text)
    return self._h_re.sub(self._h_sub, text)