import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def with_defaults_from_gate(self, gate: 'cirq.Gate', gate_to_angles_func=phased_fsim_angles_from_gate):
    """A new Options class with {angle}_defaults inferred from `gate`.

        This keeps the same settings for the characterize_{angle} booleans, but will disregard
        any current {angle}_default values.
        """
    return XEBPhasedFSimCharacterizationOptions(characterize_theta=self.characterize_theta, characterize_zeta=self.characterize_zeta, characterize_chi=self.characterize_chi, characterize_gamma=self.characterize_gamma, characterize_phi=self.characterize_phi, **gate_to_angles_func(gate))