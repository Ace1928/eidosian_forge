from typing import Any, List
import abc
import sympy
import cirq
from cirq_google.api import v2
from cirq_google.serialization import arg_func_langs
Turns a cirq.google.api.v2.CircuitOperation proto into a CircuitOperation.

        Args:
            proto: The proto object to be deserialized.
            arg_function_language: The `arg_function_language` field from
                `Program.Language`.
            constants: The list of Constant protos referenced by constant
                table indices in `proto`. This list should already have been
                parsed to produce 'deserialized_constants'.
            deserialized_constants: The deserialized contents of `constants`.

        Returns:
            The deserialized CircuitOperation represented by `proto`.

        Raises:
            ValueError: If the circuit operatio proto cannot be deserialied because it is malformed.
        