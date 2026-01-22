import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _xml_oneliner_re_from_tab_width(tab_width):
    """Standalone XML processing instruction regex."""
    return re.compile('\n        (?:\n            (?<=\\n\\n)       # Starting after a blank line\n            |               # or\n            \\A\\n?           # the beginning of the doc\n        )\n        (                           # save in $1\n            [ ]{0,%d}\n            (?:\n                <\\?\\w+\\b\\s+.*?\\?>   # XML processing instruction\n                |\n                <\\w+:\\w+\\b\\s+.*?/>  # namespaced single tag\n            )\n            [ \\t]*\n            (?=\\n{2,}|\\Z)       # followed by a blank line or end of document\n        )\n        ' % (tab_width - 1), re.X)