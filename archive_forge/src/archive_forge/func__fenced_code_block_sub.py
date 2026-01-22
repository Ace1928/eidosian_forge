import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _fenced_code_block_sub(self, match):
    return self._code_block_sub(match, is_fenced_code_block=True)