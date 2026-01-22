from pyrsistent._checked_types import CheckedType, _restore_pickle, InvariantException, store_invariants
from pyrsistent._field_common import (
from pyrsistent._pmap import PMap, pmap
class _PRecordMeta(type):

    def __new__(mcs, name, bases, dct):
        set_fields(dct, bases, name='_precord_fields')
        store_invariants(dct, bases, '_precord_invariants', '__invariant__')
        dct['_precord_mandatory_fields'] = set((name for name, field in dct['_precord_fields'].items() if field.mandatory))
        dct['_precord_initial_values'] = dict(((k, field.initial) for k, field in dct['_precord_fields'].items() if field.initial is not PFIELD_NO_INITIAL))
        dct['__slots__'] = ()
        return super(_PRecordMeta, mcs).__new__(mcs, name, bases, dct)