import contextlib
import multiprocessing
import multiprocessing.pool
from typing import Optional, Union, Iterator
import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
import cirq.experiments.xeb_fitting as xebf
import cirq.experiments.xeb_sampling as xebsamp
from cirq_google.calibration.phased_fsim import (
Run a calibration request using `cirq.experiments` XEB utilities and a sampler rather
    than `Engine.run_calibrations`.

    Args:
        calibration: A LocalXEBPhasedFSimCalibration request describing the XEB characterization
            to carry out.
        sampler: A sampler to execute circuits.
    