import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union
from qiskit import qobj, pulse
from qiskit.assembler.run_config import RunConfig
from qiskit.exceptions import QiskitError
from qiskit.pulse import instructions, transforms, library, schedule, channels
from qiskit.qobj import utils as qobj_utils, converters
from qiskit.qobj.converters.pulse_instruction import ParametricPulseShapes
def _assemble_experiments(schedules: List[Union[schedule.ScheduleComponent, Tuple[int, schedule.ScheduleComponent]]], lo_converter: converters.LoConfigConverter, run_config: RunConfig) -> Tuple[List[qobj.PulseQobjExperiment], Dict[str, Any]]:
    """Assembles a list of schedules into PulseQobjExperiments, and returns related metadata that
    will be assembled into the Qobj configuration.

    Args:
        schedules: Schedules to assemble.
        lo_converter: The configured frequency converter and validator.
        run_config: Configuration of the runtime environment.

    Returns:
        The list of assembled experiments, and the dictionary of related experiment config.

    Raises:
        QiskitError: when frequency settings are not compatible with the experiments.
    """
    freq_configs = [lo_converter(lo_dict) for lo_dict in getattr(run_config, 'schedule_los', [])]
    if len(schedules) > 1 and len(freq_configs) not in [0, 1, len(schedules)]:
        raise QiskitError("Invalid 'schedule_los' setting specified. If specified, it should be either have a single entry to apply the same LOs for each schedule or have length equal to the number of schedules.")
    instruction_converter = getattr(run_config, 'instruction_converter', converters.InstructionToQobjConverter)
    instruction_converter = instruction_converter(qobj.PulseQobjInstruction, **run_config.to_dict())
    formatted_schedules = [transforms.target_qobj_transform(sched) for sched in schedules]
    compressed_schedules = transforms.compress_pulses(formatted_schedules)
    user_pulselib = {}
    experiments = []
    for idx, sched in enumerate(compressed_schedules):
        qobj_instructions, max_memory_slot = _assemble_instructions(sched, instruction_converter, run_config, user_pulselib)
        metadata = sched.metadata
        if metadata is None:
            metadata = {}
        qobj_experiment_header = qobj.QobjExperimentHeader(memory_slots=max_memory_slot + 1, name=sched.name or 'Experiment-%d' % idx, metadata=metadata)
        experiment = qobj.PulseQobjExperiment(header=qobj_experiment_header, instructions=qobj_instructions)
        if freq_configs:
            freq_idx = idx if len(freq_configs) != 1 else 0
            experiment.config = freq_configs[freq_idx]
        experiments.append(experiment)
    if freq_configs and len(experiments) == 1:
        experiment = experiments[0]
        experiments = []
        for freq_config in freq_configs:
            experiments.append(qobj.PulseQobjExperiment(header=experiment.header, instructions=experiment.instructions, config=freq_config))
    experiment_config = {'pulse_library': [qobj.PulseLibraryItem(name=name, samples=samples) for name, samples in user_pulselib.items()], 'memory_slots': max((exp.header.memory_slots for exp in experiments))}
    return (experiments, experiment_config)