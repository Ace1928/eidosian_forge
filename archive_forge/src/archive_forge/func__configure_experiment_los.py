import copy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from qiskit.assembler.run_config import RunConfig
from qiskit.assembler.assemble_schedules import _assemble_instructions as _assemble_schedule
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.classicalregister import Clbit
from qiskit.exceptions import QiskitError
from qiskit.qobj import (
from qiskit.utils.parallel import parallel_map
def _configure_experiment_los(experiments: List[QasmQobjExperiment], lo_converter: converters.LoConfigConverter, run_config: RunConfig):
    freq_configs = [lo_converter(lo_dict) for lo_dict in getattr(run_config, 'schedule_los', [])]
    if len(experiments) > 1 and len(freq_configs) not in [0, 1, len(experiments)]:
        raise QiskitError("Invalid 'schedule_los' setting specified. If specified, it should be either have a single entry to apply the same LOs for each experiment or have length equal to the number of experiments.")
    if len(freq_configs) > 1:
        if len(experiments) > 1:
            for idx, expt in enumerate(experiments):
                freq_config = freq_configs[idx]
                expt.config.qubit_lo_freq = freq_config.qubit_lo_freq
                expt.config.meas_lo_freq = freq_config.meas_lo_freq
        elif len(experiments) == 1:
            expt = experiments[0]
            experiments = []
            for freq_config in freq_configs:
                expt_config = copy.deepcopy(expt.config)
                expt_config.qubit_lo_freq = freq_config.qubit_lo_freq
                expt_config.meas_lo_freq = freq_config.meas_lo_freq
                experiments.append(QasmQobjExperiment(header=expt.header, instructions=expt.instructions, config=expt_config))
    return experiments