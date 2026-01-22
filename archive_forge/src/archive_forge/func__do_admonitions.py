import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _do_admonitions(self, text):
    return self._admonitions_re.sub(self._do_admonitions_sub, text)