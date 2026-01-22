from . import screen
from . import FSM
import string
def DoHomeOrigin(fsm):
    c = 1
    r = 1
    screen = fsm.memory[0]
    screen.cursor_home(r, c)