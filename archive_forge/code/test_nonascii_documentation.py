import os
import sys
from unicodedata import normalize
from .. import osutils
from ..osutils import pathjoin
from . import TestCase, TestCaseWithTransport, TestSkipped
Test adding in a non-ASCII branch.