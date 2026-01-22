import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union
from qiskit import qobj, pulse
from qiskit.assembler.run_config import RunConfig
from qiskit.exceptions import QiskitError
from qiskit.pulse import instructions, transforms, library, schedule, channels
from qiskit.qobj import utils as qobj_utils, converters
from qiskit.qobj.converters.pulse_instruction import ParametricPulseShapes
Assembles the QobjConfiguration from experimental config and runtime config.

    Args:
        lo_converter: The configured frequency converter and validator.
        experiment_config: Schedules to assemble.
        run_config: Configuration of the runtime environment.

    Returns:
        The assembled PulseQobjConfig.
    