import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _header_id_exists(self, text):
    header_id = _slugify(text)
    prefix = self.extras['header-ids'].get('prefix')
    if prefix and isinstance(prefix, str):
        header_id = prefix + '-' + header_id
    return header_id in self._count_from_header_id or header_id in map(lambda x: x[1], self._toc)