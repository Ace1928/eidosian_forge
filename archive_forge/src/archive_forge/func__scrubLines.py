from unittest import TestCase
from ipykernel.tests import utils
from nbformat.converter import convert
from nbformat.reader import reads
import re
import json
from copy import copy
import unittest
def _scrubLines(lines):
    if isstr(lines):
        return re.sub(scrub, sub, lines)
    else:
        return [re.sub(scrub, sub, line) for line in lines]