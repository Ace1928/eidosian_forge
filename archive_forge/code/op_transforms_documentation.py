import functools
import inspect
import os
import warnings
import pennylane as qml
Given an input object, which may be:

        - an object such as a tape or a operation, or
        - a callable such as a QNode or a quantum function
          (alongside the callable arguments ``args`` and ``kwargs``),

        this function constructs and returns the tape/operation
        represented by the object.

        The ``wire_order`` argument determines whether a custom wire ordering
        should be used. If not provided, the wire ordering defaults to the
        objects wire ordering accessed via ``obj.wires``.

        Returns:
            tuple[.QuantumTape, Wires]: returns the tape and the verified wire order
        