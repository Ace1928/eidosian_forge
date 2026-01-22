import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _block_quote_sub(self, match):
    bq = match.group(1)
    is_spoiler = 'spoiler' in self.extras and self._bq_all_lines_spoilers.match(bq)
    if is_spoiler:
        bq = self._bq_one_level_re_spoiler.sub('', bq)
    else:
        bq = self._bq_one_level_re.sub('', bq)
    bq = self._ws_only_line_re.sub('', bq)
    bq = self._run_block_gamut(bq)
    bq = re.sub('(?m)^', '  ', bq)
    bq = self._html_pre_block_re.sub(self._dedent_two_spaces_sub, bq)
    if is_spoiler:
        return '<blockquote class="spoiler">\n%s\n</blockquote>\n\n' % bq
    else:
        return '<blockquote>\n%s\n</blockquote>\n\n' % bq