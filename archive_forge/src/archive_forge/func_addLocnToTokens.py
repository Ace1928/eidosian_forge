from pyparsing import *
def addLocnToTokens(s, l, t):
    t['locn'] = l
    t['word'] = t[0]