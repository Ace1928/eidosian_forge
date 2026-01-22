from . import screen
from . import FSM
import string
def DoMode(fsm):
    screen = fsm.memory[0]
    mode = fsm.memory.pop()