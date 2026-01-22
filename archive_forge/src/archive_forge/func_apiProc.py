from pyparsing import *
def apiProc(name, numargs):
    return LBRACK + Keyword(name)('procname') - arg * numargs + RBRACK