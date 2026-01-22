import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _add_newline(self, inner):
    yield (0, '\n')
    yield from inner
    yield (0, '\n')