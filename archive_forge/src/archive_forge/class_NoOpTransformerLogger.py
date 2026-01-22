import dataclasses
import inspect
import enum
import functools
import textwrap
from typing import (
from typing_extensions import Protocol
from cirq import circuits
class NoOpTransformerLogger(TransformerLogger):
    """All calls to this logger are a no-op"""

    def register_initial(self, circuit: 'cirq.AbstractCircuit', transformer_name: str) -> None:
        pass

    def log(self, *args: str, level: LogLevel=LogLevel.INFO) -> None:
        pass

    def register_final(self, circuit: 'cirq.AbstractCircuit', transformer_name: str) -> None:
        pass

    def show(self, level: LogLevel=LogLevel.INFO) -> None:
        pass