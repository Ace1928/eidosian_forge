import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
@staticmethod
def _uniform_indent(text, indent, include_empty_lines=False, indent_empty_lines=False):
    """
        Uniformly indent a block of text by a fixed amount

        Args:
            text: the text to indent
            indent: a string containing the indent to apply
            include_empty_lines: don't remove whitespace only lines
            indent_empty_lines: indent whitespace only lines with the rest of the text
        """
    blocks = []
    for line in text.splitlines(True):
        if line.strip() or indent_empty_lines:
            blocks.append(indent + line)
        elif include_empty_lines:
            blocks.append(line)
        else:
            blocks.append('')
    return ''.join(blocks)