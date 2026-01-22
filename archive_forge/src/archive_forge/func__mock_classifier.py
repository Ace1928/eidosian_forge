import itertools
from typing import TYPE_CHECKING, Type, Callable, Dict, Optional, Union, Iterable, Sequence, List
from cirq import ops, circuits, protocols, _import
from cirq.transformers import transformer_api
def _mock_classifier(op: 'cirq.Operation') -> bool:
    """Mock classifier, used to "complete" a collection of classifiers and make it exhaustive."""
    return False