from collections import namedtuple
from collections.abc import Iterable, Sized
from html import escape as htmlescape
from itertools import chain, zip_longest as izip_longest
from functools import reduce, partial
import io
import re
import math
import textwrap
import dataclasses
def _pprint_file(fobject, headers, tablefmt, sep, floatfmt, intfmt, file, colalign):
    rows = fobject.readlines()
    table = [re.split(sep, r.rstrip()) for r in rows if r.strip()]
    print(tabulate(table, headers, tablefmt, floatfmt=floatfmt, intfmt=intfmt, colalign=colalign), file=file)