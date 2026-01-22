import re
from lib2to3.fixer_util import Leaf, Node, Comma
from lib2to3 import fixer_base
from libfuturize.fixer_util import (token, future_import, touch_import_top,

        Since the tree needs to be fixed once and only once if and only if it
        matches, we can start discarding matches after the first.
        