import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _do_strike(self, text):
    text = self._strike_re.sub('<s>\\1</s>', text)
    return text