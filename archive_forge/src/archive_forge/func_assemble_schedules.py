import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union
from qiskit import qobj, pulse
from qiskit.assembler.run_config import RunConfig
from qiskit.exceptions import QiskitError
from qiskit.pulse import instructions, transforms, library, schedule, channels
from qiskit.qobj import utils as qobj_utils, converters
from qiskit.qobj.converters.pulse_instruction import ParametricPulseShapes
def assemble_schedules(schedules: List[Union[schedule.ScheduleBlock, schedule.ScheduleComponent, Tuple[int, schedule.ScheduleComponent]]], qobj_id: int, qobj_header: qobj.QobjHeader, run_config: RunConfig) -> qobj.PulseQobj:
    """Assembles a list of schedules into a qobj that can be run on the backend.

    Args:
        schedules: Schedules to assemble.
        qobj_id: Identifier for the generated qobj.
        qobj_header: Header to pass to the results.
        run_config: Configuration of the runtime environment.

    Returns:
        The Qobj to be run on the backends.

    Raises:
        QiskitError: when frequency settings are not supplied.

    Examples:

        .. code-block:: python

            from qiskit import pulse
            from qiskit.assembler import assemble_schedules
            from qiskit.assembler.run_config import RunConfig
            # Construct a Qobj header for the output Qobj
            header = {"backend_name": "FakeOpenPulse2Q", "backend_version": "0.0.0"}
            # Build a configuration object for the output Qobj
            config = RunConfig(shots=1024,
                               memory=False,
                               meas_level=1,
                               meas_return='avg',
                               memory_slot_size=100,
                               parametric_pulses=[],
                               init_qubits=True,
                               qubit_lo_freq=[4900000000.0, 5000000000.0],
                               meas_lo_freq=[6500000000.0, 6600000000.0],
                               schedule_los=[])
            # Build a Pulse schedule to assemble into a Qobj
            schedule = pulse.Schedule()
            schedule += pulse.Play(pulse.Waveform([0.1] * 16, name="test0"),
                                   pulse.DriveChannel(0),
                                   name="test1")
            schedule += pulse.Play(pulse.Waveform([0.1] * 16, name="test1"),
                                   pulse.DriveChannel(0),
                                   name="test2")
            schedule += pulse.Play(pulse.Waveform([0.5] * 16, name="test0"),
                                   pulse.DriveChannel(0),
                                   name="test1")
            # Assemble a Qobj from the schedule.
            pulseQobj = assemble_schedules(schedules=[schedule],
                                           qobj_id="custom-id",
                                           qobj_header=header,
                                           run_config=config)
    """
    if not hasattr(run_config, 'qubit_lo_freq'):
        raise QiskitError('qubit_lo_freq must be supplied.')
    if not hasattr(run_config, 'meas_lo_freq'):
        raise QiskitError('meas_lo_freq must be supplied.')
    lo_converter = converters.LoConfigConverter(qobj.PulseQobjExperimentConfig, **run_config.to_dict())
    experiments, experiment_config = _assemble_experiments(schedules, lo_converter, run_config)
    qobj_config = _assemble_config(lo_converter, experiment_config, run_config)
    return qobj.PulseQobj(experiments=experiments, qobj_id=qobj_id, header=qobj_header, config=qobj_config)