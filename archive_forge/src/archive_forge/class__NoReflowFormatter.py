import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
class _NoReflowFormatter(argparse.RawDescriptionHelpFormatter):
    """An argparse formatter that does NOT reflow the description."""

    def format_description(self, description):
        return description or ''