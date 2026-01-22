import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _list_item_sub(self, match):
    item = match.group(4)
    leading_line = match.group(1)
    if leading_line or '\n\n' in item or self._last_li_endswith_two_eols:
        item = self._uniform_outdent(item, min_outdent=' ', max_outdent=self.tab)[1]
        item = self._run_block_gamut(item)
    else:
        item = self._do_lists(self._uniform_outdent(item, min_outdent=' ')[1])
        if item.endswith('\n'):
            item = item[:-1]
        item = self._run_span_gamut(item)
    self._last_li_endswith_two_eols = len(match.group(5)) == 2
    if 'task_list' in self.extras:
        item = self._task_list_item_re.sub(self._task_list_item_sub, item)
    return '<li>%s</li>\n' % item