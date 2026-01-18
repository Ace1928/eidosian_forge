import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def timespec(self):
    try:
        try:
            pass
            alt1 = 2
            LA1_0 = self.input.LA(1)
            if LA1_0 == EVERY:
                LA1_1 = self.input.LA(2)
                if DIGIT <= LA1_1 <= DIGITS:
                    alt1 = 2
                elif DAY <= LA1_1 <= SUNDAY:
                    alt1 = 1
                else:
                    nvae = NoViableAltException('', 1, 1, self.input)
                    raise nvae
            elif DIGIT <= LA1_0 <= DIGITS or FIRST <= LA1_0 <= FOURTH_OR_FIFTH:
                alt1 = 1
            else:
                nvae = NoViableAltException('', 1, 0, self.input)
                raise nvae
            if alt1 == 1:
                pass
                self._state.following.append(self.FOLLOW_specifictime_in_timespec44)
                self.specifictime()
                self._state.following.pop()
            elif alt1 == 2:
                pass
                self._state.following.append(self.FOLLOW_interval_in_timespec48)
                self.interval()
                self._state.following.pop()
            self.match(self.input, EOF, self.FOLLOW_EOF_in_timespec52)
        except RecognitionException as re:
            self.reportError(re)
            self.recover(self.input, re)
    finally:
        pass
    return