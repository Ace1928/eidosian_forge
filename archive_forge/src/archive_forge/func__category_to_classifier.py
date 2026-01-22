import itertools
from typing import TYPE_CHECKING, Type, Callable, Dict, Optional, Union, Iterable, Sequence, List
from cirq import ops, circuits, protocols, _import
from cirq.transformers import transformer_api
def _category_to_classifier(category) -> Classifier:
    """Normalizes the given category into a classifier function."""
    if isinstance(category, ops.Gate):
        return lambda op: op.gate == category
    if isinstance(category, ops.Operation):
        return lambda op: op == category
    elif isinstance(category, type) and issubclass(category, ops.Gate):
        return lambda op: isinstance(op.gate, category)
    elif isinstance(category, type) and issubclass(category, ops.Operation):
        return lambda op: isinstance(op, category)
    elif callable(category):
        return lambda op: category(op)
    else:
        raise TypeError(f'Unrecognized classifier type {type(category)} ({category!r}).\nExpected a cirq.Gate, cirq.Operation, Type[cirq.Gate], Type[cirq.Operation], or Callable[[cirq.Operation], bool].')