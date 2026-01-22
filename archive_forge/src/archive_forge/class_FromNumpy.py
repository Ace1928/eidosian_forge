import abc
from typing import TYPE_CHECKING, List, Union
from ray.data._internal.execution.interfaces import RefBundle
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data.block import Block, BlockMetadata
from ray.types import ObjectRef
class FromNumpy(AbstractFrom):
    """Logical operator for `from_numpy`."""
    pass