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
def _read_symbolic_pulse_v6(file_obj, version, use_symengine):
    make = formats.SYMBOLIC_PULSE_V2._make
    pack = formats.SYMBOLIC_PULSE_PACK_V2
    size = formats.SYMBOLIC_PULSE_SIZE_V2
    header = make(struct.unpack(pack, file_obj.read(size)))
    class_name = file_obj.read(header.class_name_size).decode(common.ENCODE)
    pulse_type = file_obj.read(header.type_size).decode(common.ENCODE)
    envelope = _loads_symbolic_expr(file_obj.read(header.envelope_size), use_symengine)
    constraints = _loads_symbolic_expr(file_obj.read(header.constraints_size), use_symengine)
    valid_amp_conditions = _loads_symbolic_expr(file_obj.read(header.valid_amp_conditions_size), use_symengine)
    parameters = common.read_mapping(file_obj, deserializer=value.loads_value, version=version, vectors={})
    duration = value.read_value(file_obj, version, {})
    name = value.read_value(file_obj, version, {})
    if class_name == 'SymbolicPulse':
        return library.SymbolicPulse(pulse_type=pulse_type, duration=duration, parameters=parameters, name=name, limit_amplitude=header.amp_limited, envelope=envelope, constraints=constraints, valid_amp_conditions=valid_amp_conditions)
    elif class_name == 'ScalableSymbolicPulse':
        if isinstance(parameters['amp'], complex):
            parameters['angle'] = np.angle(parameters['amp'])
            parameters['amp'] = np.abs(parameters['amp'])
            warnings.warn(f'ScalableSymbolicPulse with complex amp are no longer supported. {pulse_type} with complex amp was converted to (amp,angle) representation.', UserWarning)
        return library.ScalableSymbolicPulse(pulse_type=pulse_type, duration=duration, amp=parameters['amp'], angle=parameters['angle'], parameters=parameters, name=name, limit_amplitude=header.amp_limited, envelope=envelope, constraints=constraints, valid_amp_conditions=valid_amp_conditions)
    else:
        raise NotImplementedError(f"Unknown class '{class_name}'")