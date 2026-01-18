from typing import Sequence
import cirq
import cirq.work as cw
import numpy as np
This simulates asymmetric readout error.

    The noise is structured so the T1 decay is applied, then the readout bitflip, then measurement.
    If a circuit contains measurements, they must be in moments that don't also contain gates.
    