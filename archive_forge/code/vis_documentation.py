from __future__ import absolute_import, print_function, division
import locale
from itertools import islice
from collections import defaultdict
from petl.compat import numeric_types, text_type
from petl import config
from petl.util.base import Table
from petl.io.sources import MemorySource
from petl.io.html import tohtml

    Display **all rows** from a table inline within an IPython notebook (use
    with caution, big tables will kill your browser).

    