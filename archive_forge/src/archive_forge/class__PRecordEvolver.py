from pyrsistent._checked_types import CheckedType, _restore_pickle, InvariantException, store_invariants
from pyrsistent._field_common import (
from pyrsistent._pmap import PMap, pmap
class _PRecordEvolver(PMap._Evolver):
    __slots__ = ('_destination_cls', '_invariant_error_codes', '_missing_fields', '_factory_fields', '_ignore_extra')

    def __init__(self, cls, original_pmap, _factory_fields=None, _ignore_extra=False):
        super(_PRecordEvolver, self).__init__(original_pmap)
        self._destination_cls = cls
        self._invariant_error_codes = []
        self._missing_fields = []
        self._factory_fields = _factory_fields
        self._ignore_extra = _ignore_extra

    def __setitem__(self, key, original_value):
        self.set(key, original_value)

    def set(self, key, original_value):
        field = self._destination_cls._precord_fields.get(key)
        if field:
            if self._factory_fields is None or field in self._factory_fields:
                try:
                    if is_field_ignore_extra_complaint(PRecord, field, self._ignore_extra):
                        value = field.factory(original_value, ignore_extra=self._ignore_extra)
                    else:
                        value = field.factory(original_value)
                except InvariantException as e:
                    self._invariant_error_codes += e.invariant_errors
                    self._missing_fields += e.missing_fields
                    return self
            else:
                value = original_value
            check_type(self._destination_cls, field, key, value)
            is_ok, error_code = field.invariant(value)
            if not is_ok:
                self._invariant_error_codes.append(error_code)
            return super(_PRecordEvolver, self).set(key, value)
        else:
            raise AttributeError("'{0}' is not among the specified fields for {1}".format(key, self._destination_cls.__name__))

    def persistent(self):
        cls = self._destination_cls
        is_dirty = self.is_dirty()
        pm = super(_PRecordEvolver, self).persistent()
        if is_dirty or not isinstance(pm, cls):
            result = cls(_precord_buckets=pm._buckets, _precord_size=pm._size)
        else:
            result = pm
        if cls._precord_mandatory_fields:
            self._missing_fields += tuple(('{0}.{1}'.format(cls.__name__, f) for f in cls._precord_mandatory_fields - set(result.keys())))
        if self._invariant_error_codes or self._missing_fields:
            raise InvariantException(tuple(self._invariant_error_codes), tuple(self._missing_fields), 'Field invariant failed')
        check_global_invariants(result, cls._precord_invariants)
        return result