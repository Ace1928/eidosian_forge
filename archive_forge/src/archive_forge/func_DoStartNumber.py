from . import screen
from . import FSM
import string
def DoStartNumber(fsm):
    fsm.memory.append(fsm.input_symbol)