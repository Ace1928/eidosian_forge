from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import string_types, text_type
from petl.util.base import Table, iterpeek
from petl.io.numpy import construct_dtype
Append data into a bcolz ctable. The `obj` argument can be either an
    existing ctable or the name of a directory were an on-disk ctable is
    stored.

    .. versionadded:: 1.1.0

    