from . import screen
from . import FSM
import string
def DoCursorRestore(fsm):
    screen = fsm.memory[0]
    screen.cursor_restore_attrs()