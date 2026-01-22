import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _setup_extras(self):
    if 'footnotes' in self.extras:
        self.footnotes = OrderedDict()
        self.footnote_ids = []
    if 'header-ids' in self.extras:
        if not hasattr(self, '_count_from_header_id') or self.extras['header-ids'].get('reset-count', False):
            self._count_from_header_id = defaultdict(int)
    if 'metadata' in self.extras:
        self.metadata = {}