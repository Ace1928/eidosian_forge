from __future__ import annotations
import enum
from .types import Type, Bool, Uint
Determine the sort of cast that is required to move from the left type to the right type.

    Examples:

        .. code-block:: python

            >>> from qiskit.circuit.classical import types
            >>> types.cast_kind(types.Bool(), types.Bool())
            <CastKind.EQUAL: 1>
            >>> types.cast_kind(types.Uint(8), types.Bool())
            <CastKind.IMPLICIT: 2>
            >>> types.cast_kind(types.Bool(), types.Uint(8))
            <CastKind.LOSSLESS: 3>
            >>> types.cast_kind(types.Uint(16), types.Uint(8))
            <CastKind.DANGEROUS: 4>
    