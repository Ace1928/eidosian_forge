import typing
from typing import Generic
from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Group
from pennylane.data.base.mapper import AttributeTypeMapper
from pennylane.data.base.typing_util import T
class DatasetTuple(Generic[T], DatasetAttribute[HDF5Group, typing.Tuple[T], typing.Tuple[T]]):
    """Type for tuples."""
    type_id = 'tuple'

    @classmethod
    def consumes_types(cls) -> typing.Tuple[typing.Type[tuple]]:
        return (tuple,)

    @classmethod
    def default_value(cls) -> typing.Tuple[()]:
        return tuple()

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: typing.Tuple[T]) -> HDF5Group:
        grp = bind_parent.create_group(key)
        mapper = AttributeTypeMapper(grp)
        for i, elem in enumerate(value):
            mapper[str(i)] = elem
        return grp

    def hdf5_to_value(self, bind: HDF5Group) -> typing.Tuple[T]:
        mapper = AttributeTypeMapper(bind)
        return tuple((mapper[str(i)].copy_value() for i in range(len(self.bind))))