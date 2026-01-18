from __future__ import annotations
import dataclasses
import functools
import inspect
import sys
from collections import OrderedDict, defaultdict, deque, namedtuple
from operator import methodcaller
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Iterable, NamedTuple, Sequence, overload
from typing_extensions import Self  # Python 3.11+
from optree import _C
from optree.typing import (
from optree.utils import safe_zip, total_order_sorted, unzip2
def unregister_pytree_node(cls: type[CustomTreeNode[T]], *, namespace: str) -> PyTreeNodeRegistryEntry:
    """Remove a type from the pytree node registry.

    See also :func:`register_pytree_node` and :func:`register_pytree_node_class`.

    This function is the inverse operation of function :func:`register_pytree_node`.

    Args:
        cls (type): A Python type to remove from the pytree node registry.
        namespace (str): The namespace of the pytree node registry to remove the type from.

    Returns:
        The removed registry entry.

    Raises:
        TypeError: If the input type is not a class.
        TypeError: If the namespace is not a string.
        ValueError: If the namespace is an empty string.
        ValueError: If the type is a built-in type that cannot be unregistered.
        ValueError: If the type is not found in the registry.

    Examples:
        >>> # Register a Python type with lambda functions
        >>> register_pytree_node(
        ...     set,
        ...     lambda s: (sorted(s), None, None),
        ...     lambda _, children: set(children),
        ...     namespace='temp',
        ... )
        <class 'set'>

        >>> # Unregister the Python type
        >>> unregister_pytree_node(set, namespace='temp')
    """
    if not inspect.isclass(cls):
        raise TypeError(f'Expected a class, got {cls}.')
    if namespace is not __GLOBAL_NAMESPACE and (not isinstance(namespace, str)):
        raise TypeError(f'The namespace must be a string, got {namespace}.')
    if namespace == '':
        raise ValueError('The namespace cannot be an empty string.')
    registration_key: type | tuple[str, type]
    if namespace is __GLOBAL_NAMESPACE:
        registration_key = cls
        namespace = ''
    else:
        registration_key = (namespace, cls)
    with __REGISTRY_LOCK:
        _C.unregister_node(cls, namespace)
        return _NODETYPE_REGISTRY.pop(registration_key)