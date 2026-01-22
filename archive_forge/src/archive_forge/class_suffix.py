import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.deprecation import deprecated
from pyomo.core.kernel.base import ICategorizedObject, _abstract_readonly_property
from pyomo.core.kernel.dict_container import DictContainer
from pyomo.core.kernel.container_utils import define_homogeneous_container_type
class suffix(ISuffix):
    """A container for storing extraneous model data that
    can be imported to or exported from a solver."""
    _ctype = ISuffix
    __slots__ = ('_parent', '_storage_key', '_active', '_direction', '_datatype', '__weakref__')
    LOCAL = 0
    EXPORT = 1
    IMPORT = 2
    IMPORT_EXPORT = 3
    _directions = {LOCAL: 'suffix.LOCAL', EXPORT: 'suffix.EXPORT', IMPORT: 'suffix.IMPORT', IMPORT_EXPORT: 'suffix.IMPORT_EXPORT'}
    FLOAT = 4
    INT = 0
    _datatypes = {FLOAT: 'suffix.FLOAT', INT: 'suffix.INT', None: str(None)}

    def __init__(self, *args, **kwds):
        self._parent = None
        self._storage_key = None
        self._active = True
        self._direction = None
        self._datatype = None
        self.direction = kwds.pop('direction', suffix.LOCAL)
        self.datatype = kwds.pop('datatype', suffix.FLOAT)
        super(suffix, self).__init__(*args, **kwds)

    def export_enabled(self):
        """Returns :const:`True` when this suffix is enabled
        for export to solvers."""
        return bool(self._direction & suffix.EXPORT)

    def import_enabled(self):
        """Returns :const:`True` when this suffix is enabled
        for import from solutions."""
        return bool(self._direction & suffix.IMPORT)

    @property
    def datatype(self):
        """Return the suffix datatype."""
        return self._datatype

    @datatype.setter
    def datatype(self, datatype):
        """Set the suffix datatype."""
        if datatype not in self._datatypes:
            raise ValueError('Suffix datatype must be one of: %s. \nValue given: %s' % (list(self._datatypes.values()), datatype))
        self._datatype = datatype

    @property
    def direction(self):
        """Return the suffix direction."""
        return self._direction

    @direction.setter
    def direction(self, direction):
        """Set the suffix direction."""
        if not direction in self._directions:
            raise ValueError('Suffix direction must be one of: %s. \nValue given: %s' % (list(self._directions.values()), direction))
        self._direction = direction

    @deprecated('suffix.set_all_values will be removed in the future.', version='5.3')
    def set_all_values(self, value):
        for ndx in self:
            self[ndx] = value

    @deprecated("suffix.clear_value will be removed in the future. Use 'del suffix[key]' instead.", version='5.3')
    def clear_value(self, component):
        try:
            del self[component]
        except KeyError:
            pass

    @deprecated('suffix.clear_all_values is replaced with suffix.clear', version='5.3')
    def clear_all_values(self):
        self.clear()

    @deprecated('suffix.get_datatype is replaced with the property suffix.datatype', version='5.3')
    def get_datatype(self):
        return self.datatype

    @deprecated('suffix.set_datatype is replaced with the property setter suffix.datatype', version='5.3')
    def set_datatype(self, datatype):
        self.datatype = datatype

    @deprecated('suffix.get_direction is replaced with the property suffix.direction', version='5.3')
    def get_direction(self):
        return self.direction

    @deprecated('suffix.set_direction is replaced with the property setter suffix.direction', version='5.3')
    def set_direction(self, direction):
        self.direction = direction