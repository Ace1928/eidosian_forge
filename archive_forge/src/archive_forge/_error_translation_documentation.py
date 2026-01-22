import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN

    Extract a dataset name from the given snapshot or bookmark name.

    '@' separates a snapshot name from the rest of the dataset name.
    '#' separates a bookmark name from the rest of the dataset name.
    