import types
from collections import namedtuple
from copy import deepcopy
from weakref import ref as _weakref_ref
@staticmethod
def weakref_sequence_mapper(encode, val):
    """__autoslot_mappers__ mapper for fields with sequences of weakrefs

        This mapper expects to be passed a field that is a sequence of
        weakrefs.  It will resolve all weakrefs when generating a state,
        and then convert the hard references back to a weakref when
        restoring the state.

        """
    if val is None:
        return val
    if encode:
        return val.__class__((v() for v in val))
    else:
        return val.__class__((_weakref_ref(v) for v in val))