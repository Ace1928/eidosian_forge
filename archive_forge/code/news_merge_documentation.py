import patiencediff
from merge3 import Merge3
from ... import merge
from .parser import simple_parse_lines
Perform a simple 3-way merge of a bzr NEWS file.

        Each section of a bzr NEWS file is essentially an ordered set of bullet
        points, so we can simply take a set of bullet points, determine which
        bullets to add and which to remove, sort, and reserialize.
        