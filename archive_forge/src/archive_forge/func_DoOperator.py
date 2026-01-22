import sys
import string
def DoOperator(fsm):
    ar = fsm.memory.pop()
    al = fsm.memory.pop()
    if fsm.input_symbol == '+':
        fsm.memory.append(al + ar)
    elif fsm.input_symbol == '-':
        fsm.memory.append(al - ar)
    elif fsm.input_symbol == '*':
        fsm.memory.append(al * ar)
    elif fsm.input_symbol == '/':
        fsm.memory.append(al / ar)