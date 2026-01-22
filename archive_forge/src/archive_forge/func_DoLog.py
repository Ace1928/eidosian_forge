from . import screen
from . import FSM
import string
def DoLog(fsm):
    screen = fsm.memory[0]
    fsm.memory = [screen]
    fout = open('log', 'a')
    fout.write(fsm.input_symbol + ',' + fsm.current_state + '\n')
    fout.close()