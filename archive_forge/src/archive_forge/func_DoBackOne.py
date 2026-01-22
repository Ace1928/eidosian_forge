from . import screen
from . import FSM
import string
def DoBackOne(fsm):
    screen = fsm.memory[0]
    screen.cursor_back()