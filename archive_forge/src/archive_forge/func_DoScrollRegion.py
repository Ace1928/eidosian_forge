from . import screen
from . import FSM
import string
def DoScrollRegion(fsm):
    screen = fsm.memory[0]
    r2 = int(fsm.memory.pop())
    r1 = int(fsm.memory.pop())
    screen.scroll_screen_rows(r1, r2)