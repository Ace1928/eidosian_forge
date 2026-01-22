from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from .base_result import _BasePrimitiveResult
Result of Estimator.

    .. code-block:: python

        result = estimator.run(circuits, observables, params).result()

    where the i-th elements of ``result`` correspond to the circuit and observable given by
    ``circuits[i]``, ``observables[i]``, and the parameter values bounds by ``params[i]``.
    For example, ``results.values[i]`` gives the expectation value, and ``result.metadata[i]``
    is a metadata dictionary for this circuit and parameters.

    Args:
        values (np.ndarray): The array of the expectation values.
        metadata (list[dict]): List of the metadata.
    