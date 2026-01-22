import warnings
from typing import Dict, List, Union, Optional, Set, cast, Iterable, Sequence, Tuple
import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._qvm import (
from pyquil.api._qvm_client import (
from pyquil.gates import MOVE
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program, percolate_declares
from pyquil.quilatom import MemoryReference
from pyquil.wavefunction import Wavefunction

        Run a Quil program once to determine the final wavefunction, and measure multiple times.

        Alternatively, consider using ``wavefunction`` and calling ``sample_bitstrings`` on the
        resulting object.

        For a large wavefunction and a low-medium number of trials, use this function.
        On the other hand, if you're sampling a small system many times you might want to
        use ``Wavefunction.sample_bitstrings``.

        .. note:: If your program contains measurements or noisy gates, this method may not do what
            you want. If the execution of ``quil_program`` is **non-deterministic** then the
            final wavefunction from which the returned bitstrings are sampled itself only
            represents a stochastically generated sample and the outcomes sampled from
            *different* ``run_and_measure`` calls *generally sample different bitstring
            distributions*.

        :param quil_program: The program to run and measure
        :param qubits: An optional list of qubits to measure. The order of this list is
            respected in the returned bitstrings. If not provided, all qubits used in
            the program will be measured and returned in their sorted order.
        :param int trials: Number of times to sample from the prepared wavefunction.
        :param memory_map: An assignment of classical registers to values, representing an initial
                           state for the QAM's classical memory.

                           This is expected to be of type Dict[str, List[Union[int, float]]],
                           where the keys are memory region names and the values are arrays of
                           initialization data.
        :return: An array of measurement results (0 or 1) of shape (trials, len(qubits))
        