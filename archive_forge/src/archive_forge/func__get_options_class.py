from pyarrow._compute import (  # noqa
from collections import namedtuple
import inspect
from textwrap import dedent
import warnings
import pyarrow as pa
from pyarrow import _compute_docstrings
from pyarrow.vendored import docscrape
def _get_options_class(func):
    class_name = func._doc.options_class
    if not class_name:
        return None
    try:
        return globals()[class_name]
    except KeyError:
        warnings.warn('Python binding for {} not exposed'.format(class_name), RuntimeWarning)
        return None