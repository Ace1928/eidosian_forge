from typing import Dict, Union, List
import dateutil.parser
def _decode_pulse_qobj_instr(pulse_qobj_instr: Dict) -> None:
    """Decode a pulse Qobj instruction.

    Args:
        pulse_qobj_instr: A ``PulseQobjInstruction`` in dictionary format.
    """
    if 'val' in pulse_qobj_instr:
        pulse_qobj_instr['val'] = _to_complex(pulse_qobj_instr['val'])
    if 'parameters' in pulse_qobj_instr and 'amp' in pulse_qobj_instr['parameters']:
        pulse_qobj_instr['parameters']['amp'] = _to_complex(pulse_qobj_instr['parameters']['amp'])