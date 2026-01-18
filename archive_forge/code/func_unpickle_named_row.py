import functools
from collections import namedtuple
def unpickle_named_row(names, values):
    return create_namedtuple_class(*names)(*values)