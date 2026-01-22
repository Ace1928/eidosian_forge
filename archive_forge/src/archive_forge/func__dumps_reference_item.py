import json
import struct
import zlib
import warnings
from io import BytesIO
import numpy as np
import symengine as sym
from symengine.lib.symengine_wrapper import (  # pylint: disable = no-name-in-module
from qiskit.exceptions import QiskitError
from qiskit.pulse import library, channels, instructions
from qiskit.pulse.schedule import ScheduleBlock
from qiskit.qpy import formats, common, type_keys
from qiskit.qpy.binary_io import value
from qiskit.qpy.exceptions import QpyError
from qiskit.pulse.configuration import Kernel, Discriminator
def _dumps_reference_item(schedule, metadata_serializer):
    if schedule is None:
        type_key = type_keys.Value.NULL
        data_bytes = b''
    else:
        type_key = type_keys.Program.SCHEDULE_BLOCK
        data_bytes = common.data_to_binary(obj=schedule, serializer=write_schedule_block, metadata_serializer=metadata_serializer)
    return (type_key, data_bytes)