from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.pulse.channels import Channel
from qiskit.pulse.exceptions import PulseError
Return a new schedule which is shifted forward by `time`.

        Returns:
            Schedule: The shifted schedule
        