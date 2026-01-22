import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
@staticmethod
def _match_overlaps_substr(text, match, substr):
    """
        Checks if a regex match overlaps with a substring in the given text.
        """
    for instance in re.finditer(re.escape(substr), text):
        start, end = instance.span()
        if start <= match.start() <= end:
            return True
        if start <= match.end() <= end:
            return True
    return False