from dissononce.extras.processing.handshakestate_forwarder import ForwarderHandshakeState
from transitions import Machine
from transitions.core import MachineError
import logging
def write_message(self, payload, message_buffer):
    try:
        self._handshake_machine.next_message()
        self._pattern_machine.write()
    except MachineError as ex:
        raise self._convert_machine_error(ex, 'write_message')
    result = self._handshakestate.write_message(payload, message_buffer)
    if result is not None:
        self._handshake_machine.finish()
    return result