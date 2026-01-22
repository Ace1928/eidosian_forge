import sys
from antlr3 import *
from antlr3.compat import set, frozenset
class DFA26(DFA):

    def specialStateTransition(self_, s, input):
        self = self_.recognizer
        _s = s
        if s == 0:
            LA26_0 = input.LA(1)
            s = -1
            if LA26_0 == 48:
                s = 1
            elif LA26_0 == 49:
                s = 2
            elif LA26_0 == 50:
                s = 3
            elif LA26_0 == 51:
                s = 4
            elif LA26_0 == 115:
                s = 5
            elif LA26_0 == 102:
                s = 6
            elif LA26_0 == 52:
                s = 7
            elif LA26_0 == 116:
                s = 8
            elif LA26_0 == 53:
                s = 9
            elif 54 <= LA26_0 <= 57:
                s = 10
            elif LA26_0 == 100:
                s = 11
            elif LA26_0 == 109:
                s = 12
            elif LA26_0 == 119:
                s = 13
            elif LA26_0 == 106:
                s = 14
            elif LA26_0 == 97:
                s = 15
            elif LA26_0 == 111:
                s = 16
            elif LA26_0 == 110:
                s = 17
            elif LA26_0 == 113:
                s = 18
            elif LA26_0 == 101:
                s = 19
            elif LA26_0 == 104:
                s = 20
            elif LA26_0 == 44:
                s = 21
            elif 9 <= LA26_0 <= 10 or LA26_0 == 13 or LA26_0 == 32:
                s = 22
            elif 0 <= LA26_0 <= 8 or 11 <= LA26_0 <= 12 or 14 <= LA26_0 <= 31 or (33 <= LA26_0 <= 43) or (45 <= LA26_0 <= 47) or (58 <= LA26_0 <= 96) or (98 <= LA26_0 <= 99) or (LA26_0 == 103) or (LA26_0 == 105) or (107 <= LA26_0 <= 108) or (LA26_0 == 112) or (LA26_0 == 114) or (117 <= LA26_0 <= 118) or (120 <= LA26_0 <= 65535):
                s = 23
            if s >= 0:
                return s
        if self._state.backtracking > 0:
            raise BacktrackingFailed
        nvae = NoViableAltException(self_.getDescription(), 26, _s, input)
        self_.error(nvae)
        raise nvae