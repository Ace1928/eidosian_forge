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
def _read_waveform(file_obj, version):
    header = formats.WAVEFORM._make(struct.unpack(formats.WAVEFORM_PACK, file_obj.read(formats.WAVEFORM_SIZE)))
    samples_raw = file_obj.read(header.data_size)
    samples = common.data_from_binary(samples_raw, np.load)
    name = value.read_value(file_obj, version, {})
    return library.Waveform(samples=samples, name=name, epsilon=header.epsilon, limit_amplitude=header.amp_limited)