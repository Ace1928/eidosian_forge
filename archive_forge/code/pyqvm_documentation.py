import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Type, Union
import numpy as np
from numpy.random.mtrand import RandomState
from pyquil.api import QAM, QuantumExecutable, QAMExecutionResult
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.quil import Program
from pyquil.quilatom import Label, LabelPlaceholder, MemoryReference
from pyquil.quilbase import (

        Execute one outer loop of a program on the PyQVM without re-initializing its state.

        Note that the PyQVM is stateful. Subsequent calls to :py:func:`execute_once` will not
        automatically reset the wavefunction or the classical RAM. If this is desired,
        consider starting your program with ``RESET``.

        :return: ``self`` to support method chaining.
        