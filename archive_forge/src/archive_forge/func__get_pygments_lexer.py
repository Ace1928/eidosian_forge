import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _get_pygments_lexer(self, lexer_name):
    try:
        from pygments import lexers, util
    except ImportError:
        return None
    try:
        return lexers.get_lexer_by_name(lexer_name)
    except util.ClassNotFound:
        return None