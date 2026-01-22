from . import screen
from . import FSM
import string
def DoUpOne(fsm):
    screen = fsm.memory[0]
    screen.cursor_up()