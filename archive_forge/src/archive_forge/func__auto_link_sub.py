import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _auto_link_sub(self, match):
    g1 = match.group(1)
    return '<a href="%s">%s</a>' % (self._protect_url(g1), g1)