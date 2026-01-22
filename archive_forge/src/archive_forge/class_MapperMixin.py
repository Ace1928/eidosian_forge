import typing
from collections.abc import MutableMapping
from types import MappingProxyType
from typing import Any, Dict, Optional, Type
from pennylane.data.base.attribute import (
from pennylane.data.base.hdf5 import HDF5Any, HDF5Group
class MapperMixin:
    """Mixin class for Dataset types that provide an interface
    to a HDF5 group, e.g `DatasetList`, `DatasetDict`. Provides
    a `_mapper` property over the type's ``bind`` attribute."""
    bind: HDF5Group
    __mapper: AttributeTypeMapper = None

    @property
    def _mapper(self) -> AttributeTypeMapper:
        if self.__mapper is None:
            self.__mapper = AttributeTypeMapper(self.bind)
        return self.__mapper