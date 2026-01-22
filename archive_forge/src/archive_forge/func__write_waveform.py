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
def _write_waveform(file_obj, data):
    samples_bytes = common.data_to_binary(data.samples, np.save)
    header = struct.pack(formats.WAVEFORM_PACK, data.epsilon, len(samples_bytes), data._limit_amplitude)
    file_obj.write(header)
    file_obj.write(samples_bytes)
    value.write_value(file_obj, data.name)