import functools
import warnings
from typing import (
import torch
import optree
from optree import PyTreeSpec  # direct import for type annotations
def _private_register_pytree_node(cls: Type[Any], flatten_fn: FlattenFunc, unflatten_fn: UnflattenFunc, *, serialized_type_name: Optional[str]=None, to_dumpable_context: Optional[ToDumpableContextFn]=None, from_dumpable_context: Optional[FromDumpableContextFn]=None) -> None:
    """This is an internal function that is used to register a pytree node type
    for the C++ pytree only. End-users should use :func:`register_pytree_node`
    instead.
    """
    if not optree.is_structseq_class(cls):
        optree.register_pytree_node(cls, flatten_fn, _reverse_args(unflatten_fn), namespace='torch')