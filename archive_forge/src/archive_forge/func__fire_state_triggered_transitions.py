from typing import cast, Dict, Optional, Set, Tuple, Type, Union
from ._events import *
from ._util import LocalProtocolError, Sentinel
def _fire_state_triggered_transitions(self) -> None:
    while True:
        start_states = dict(self.states)
        if self.pending_switch_proposals:
            if self.states[CLIENT] is DONE:
                self.states[CLIENT] = MIGHT_SWITCH_PROTOCOL
        if not self.pending_switch_proposals:
            if self.states[CLIENT] is MIGHT_SWITCH_PROTOCOL:
                self.states[CLIENT] = DONE
        if not self.keep_alive:
            for role in (CLIENT, SERVER):
                if self.states[role] is DONE:
                    self.states[role] = MUST_CLOSE
        joint_state = (self.states[CLIENT], self.states[SERVER])
        changes = STATE_TRIGGERED_TRANSITIONS.get(joint_state, {})
        self.states.update(changes)
        if self.states == start_states:
            return