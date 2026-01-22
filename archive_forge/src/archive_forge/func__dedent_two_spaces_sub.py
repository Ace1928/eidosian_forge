import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _dedent_two_spaces_sub(self, match):
    return re.sub('(?m)^  ', '', match.group(1))