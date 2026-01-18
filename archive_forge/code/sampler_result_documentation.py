from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from qiskit.result import QuasiDistribution
from .base_result import _BasePrimitiveResult
Result of Sampler.

    .. code-block:: python

        result = sampler.run(circuits, params).result()

    where the i-th elements of ``result`` correspond to the circuit given by ``circuits[i]``,
    and the parameter values bounds by ``params[i]``.
    For example, ``results.quasi_dists[i]`` gives the quasi-probabilities of bitstrings, and
    ``result.metadata[i]`` is a metadata dictionary for this circuit and parameters.

    Args:
        quasi_dists (list[QuasiDistribution]): List of the quasi-probabilities.
        metadata (list[dict]): List of the metadata.
    