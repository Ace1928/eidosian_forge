import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _process_list_items(self, list_str):
    self.list_level += 1
    self._last_li_endswith_two_eols = False
    list_str = list_str.rstrip('\n') + '\n'
    list_str = self._list_item_re.sub(self._list_item_sub, list_str)
    self.list_level -= 1
    return list_str