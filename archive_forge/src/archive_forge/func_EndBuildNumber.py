import sys
import string
def EndBuildNumber(fsm):
    s = fsm.memory.pop()
    fsm.memory.append(int(s))