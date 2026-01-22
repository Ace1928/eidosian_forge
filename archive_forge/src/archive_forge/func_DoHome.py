from . import screen
from . import FSM
import string
def DoHome(fsm):
    c = int(fsm.memory.pop())
    r = int(fsm.memory.pop())
    screen = fsm.memory[0]
    screen.cursor_home(r, c)