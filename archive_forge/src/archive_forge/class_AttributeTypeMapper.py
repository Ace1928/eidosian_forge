import typing
from collections.abc import MutableMapping
from types import MappingProxyType
from typing import Any, Dict, Optional, Type
from pennylane.data.base.attribute import (
from pennylane.data.base.hdf5 import HDF5Any, HDF5Group
class AttributeTypeMapper(MutableMapping):
    """
    This class performs the mapping between the objects contained
    in a HDF5 group and Dataset attributes.
    """
    bind: HDF5Group
    _cache: Dict[str, DatasetAttribute]

    def __init__(self, bind: HDF5Group) -> None:
        self._cache = {}
        self.bind = bind

    def __getitem__(self, key: str) -> DatasetAttribute:
        if key in self._cache:
            return self._cache[key]
        h5_obj = self.bind[key]
        attr_type = get_attribute_type(h5_obj)
        attr = attr_type(bind=h5_obj)
        self._cache[key] = attr
        return attr

    @property
    def info(self) -> AttributeInfo:
        """Return ``AttributeInfo`` for ``self.bind``."""
        return AttributeInfo(self.bind.attrs)

    def set_item(self, key: str, value: Any, info: Optional[AttributeInfo], require_type: Optional[Type[DatasetAttribute]]=None) -> None:
        """Creates or replaces attribute ``key`` with ``value``, optionally
        including ``info``.

        Args:
            key: Name of attribute in HDF5 group
            value: Attribute value, either a compatible object or an already
                initialized ``DatasetAttribute``.
            info: Extra info to attach to attribute
            require_type: Force the ``value`` to be serialized as this type.
                If ``value`` is an ``DatasetAttribute``, it must be an instance of ``require_type``.
                Otherwise, ``value`` must be serializable by ``require_type``.
        """
        try:
            if isinstance(value, DatasetAttribute):
                if require_type and (not isinstance(value, require_type)):
                    raise TypeError(f"Expected '{key}' to be of type '{require_type.__name__}', but got '{type(value).__name__}'.")
                value._set_parent(self.bind, key)
                if info:
                    info.save(value.info)
            elif require_type is not None:
                require_type(value, info, parent_and_key=(self.bind, key))
            else:
                attr_type = match_obj_type(value)
                attr_type(value, info, parent_and_key=(self.bind, key))
        except ValueError as exc:
            if exc.args[0] == 'Unable to create dataset (no write intent on file)':
                raise DatasetNotWriteableError(self.bind) from exc
            raise exc
        self._cache.pop(key, None)

    def __setitem__(self, key: str, value: Any) -> None:
        self.set_item(key, value, None)

    def move(self, src: str, dest: str) -> None:
        """Moves the attribute stored at ``src`` in ``bind`` to ``dest``."""
        self.bind.move(src, dest)
        self._cache.pop(src, None)

    def view(self) -> typing.Mapping[str, DatasetAttribute]:
        """Returns a read-only mapping of the attributes in ``bind``."""
        return MappingProxyType(self)

    def __len__(self) -> int:
        return len(self.bind)

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self.bind)

    def __contains__(self, key: str) -> bool:
        return key in self._cache or key in self.bind

    def __delitem__(self, key: str) -> None:
        self._cache.pop(key, None)
        del self.bind[key]

    def __repr__(self):
        return repr(dict(self))

    def __str__(self):
        return str(dict(self))