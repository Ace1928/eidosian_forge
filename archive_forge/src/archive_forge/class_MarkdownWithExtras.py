import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
class MarkdownWithExtras(Markdown):
    """A markdowner class that enables most extras:

    - footnotes
    - fenced-code-blocks (only highlights code if 'pygments' Python module on path)

    These are not included:
    - pyshell (specific to Python-related documenting)
    - code-friendly (because it *disables* part of the syntax)
    - link-patterns (because you need to specify some actual
      link-patterns anyway)
    """
    extras = ['footnotes', 'fenced-code-blocks']