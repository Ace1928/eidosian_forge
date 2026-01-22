import sys
from antlr3 import *
from antlr3.compat import set, frozenset
class GrocLexer(Lexer):
    grammarFileName = 'borg/borgcron/py/Groc.g'
    antlr_version = version_str_to_tuple('3.1.1')
    antlr_version_str = '3.1.1'

    def __init__(self, input=None, state=None):
        if state is None:
            state = RecognizerSharedState()
        Lexer.__init__(self, input, state)
        self.dfa26 = self.DFA26(self, 26, eot=self.DFA26_eot, eof=self.DFA26_eof, min=self.DFA26_min, max=self.DFA26_max, accept=self.DFA26_accept, special=self.DFA26_special, transition=self.DFA26_transition)

    def mTIME(self):
        try:
            _type = TIME
            _channel = DEFAULT_CHANNEL
            pass
            pass
            self.mDIGIT()
            self.match(58)
            self.matchRange(48, 53)
            self.mDIGIT()
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mTWO_DIGIT_HOUR_TIME(self):
        try:
            _type = TWO_DIGIT_HOUR_TIME
            _channel = DEFAULT_CHANNEL
            pass
            pass
            alt1 = 3
            LA1 = self.input.LA(1)
            if LA1 == 48:
                alt1 = 1
            elif LA1 == 49:
                alt1 = 2
            elif LA1 == 50:
                alt1 = 3
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed
                nvae = NoViableAltException('', 1, 0, self.input)
                raise nvae
            if alt1 == 1:
                pass
                pass
                self.match(48)
                self.mDIGIT()
            elif alt1 == 2:
                pass
                pass
                self.match(49)
                self.mDIGIT()
            elif alt1 == 3:
                pass
                pass
                self.match(50)
                self.matchRange(48, 51)
            self.match(58)
            pass
            self.matchRange(48, 53)
            self.mDIGIT()
            if self._state.backtracking == 0:
                _type = TIME
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mSYNCHRONIZED(self):
        try:
            _type = SYNCHRONIZED
            _channel = DEFAULT_CHANNEL
            pass
            self.match('synchronized')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mFIRST(self):
        try:
            _type = FIRST
            _channel = DEFAULT_CHANNEL
            pass
            alt2 = 2
            LA2_0 = self.input.LA(1)
            if LA2_0 == 49:
                alt2 = 1
            elif LA2_0 == 102:
                alt2 = 2
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed
                nvae = NoViableAltException('', 2, 0, self.input)
                raise nvae
            if alt2 == 1:
                pass
                self.match('1st')
            elif alt2 == 2:
                pass
                self.match('first')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mSECOND(self):
        try:
            _type = SECOND
            _channel = DEFAULT_CHANNEL
            pass
            alt3 = 2
            LA3_0 = self.input.LA(1)
            if LA3_0 == 50:
                alt3 = 1
            elif LA3_0 == 115:
                alt3 = 2
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed
                nvae = NoViableAltException('', 3, 0, self.input)
                raise nvae
            if alt3 == 1:
                pass
                self.match('2nd')
            elif alt3 == 2:
                pass
                self.match('second')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mTHIRD(self):
        try:
            _type = THIRD
            _channel = DEFAULT_CHANNEL
            pass
            alt4 = 2
            LA4_0 = self.input.LA(1)
            if LA4_0 == 51:
                alt4 = 1
            elif LA4_0 == 116:
                alt4 = 2
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed
                nvae = NoViableAltException('', 4, 0, self.input)
                raise nvae
            if alt4 == 1:
                pass
                self.match('3rd')
            elif alt4 == 2:
                pass
                self.match('third')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mFOURTH(self):
        try:
            _type = FOURTH
            _channel = DEFAULT_CHANNEL
            pass
            pass
            self.match('4th')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mFIFTH(self):
        try:
            _type = FIFTH
            _channel = DEFAULT_CHANNEL
            pass
            pass
            self.match('5th')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mFOURTH_OR_FIFTH(self):
        try:
            _type = FOURTH_OR_FIFTH
            _channel = DEFAULT_CHANNEL
            pass
            alt5 = 2
            LA5_0 = self.input.LA(1)
            if LA5_0 == 102:
                LA5_1 = self.input.LA(2)
                if LA5_1 == 111:
                    alt5 = 1
                elif LA5_1 == 105:
                    alt5 = 2
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed
                    nvae = NoViableAltException('', 5, 1, self.input)
                    raise nvae
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed
                nvae = NoViableAltException('', 5, 0, self.input)
                raise nvae
            if alt5 == 1:
                pass
                pass
                self.match('fourth')
                if self._state.backtracking == 0:
                    _type = FOURTH
            elif alt5 == 2:
                pass
                pass
                self.match('fifth')
                if self._state.backtracking == 0:
                    _type = FIFTH
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mDAY(self):
        try:
            _type = DAY
            _channel = DEFAULT_CHANNEL
            pass
            self.match('day')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mMONDAY(self):
        try:
            _type = MONDAY
            _channel = DEFAULT_CHANNEL
            pass
            self.match('mon')
            alt6 = 2
            LA6_0 = self.input.LA(1)
            if LA6_0 == 100:
                alt6 = 1
            if alt6 == 1:
                pass
                self.match('day')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mTUESDAY(self):
        try:
            _type = TUESDAY
            _channel = DEFAULT_CHANNEL
            pass
            self.match('tue')
            alt7 = 2
            LA7_0 = self.input.LA(1)
            if LA7_0 == 115:
                alt7 = 1
            if alt7 == 1:
                pass
                self.match('sday')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mWEDNESDAY(self):
        try:
            _type = WEDNESDAY
            _channel = DEFAULT_CHANNEL
            pass
            self.match('wed')
            alt8 = 2
            LA8_0 = self.input.LA(1)
            if LA8_0 == 110:
                alt8 = 1
            if alt8 == 1:
                pass
                self.match('nesday')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mTHURSDAY(self):
        try:
            _type = THURSDAY
            _channel = DEFAULT_CHANNEL
            pass
            self.match('thu')
            alt9 = 2
            LA9_0 = self.input.LA(1)
            if LA9_0 == 114:
                alt9 = 1
            if alt9 == 1:
                pass
                self.match('rsday')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mFRIDAY(self):
        try:
            _type = FRIDAY
            _channel = DEFAULT_CHANNEL
            pass
            self.match('fri')
            alt10 = 2
            LA10_0 = self.input.LA(1)
            if LA10_0 == 100:
                alt10 = 1
            if alt10 == 1:
                pass
                self.match('day')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mSATURDAY(self):
        try:
            _type = SATURDAY
            _channel = DEFAULT_CHANNEL
            pass
            self.match('sat')
            alt11 = 2
            LA11_0 = self.input.LA(1)
            if LA11_0 == 117:
                alt11 = 1
            if alt11 == 1:
                pass
                self.match('urday')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mSUNDAY(self):
        try:
            _type = SUNDAY
            _channel = DEFAULT_CHANNEL
            pass
            self.match('sun')
            alt12 = 2
            LA12_0 = self.input.LA(1)
            if LA12_0 == 100:
                alt12 = 1
            if alt12 == 1:
                pass
                self.match('day')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mJANUARY(self):
        try:
            _type = JANUARY
            _channel = DEFAULT_CHANNEL
            pass
            self.match('jan')
            alt13 = 2
            LA13_0 = self.input.LA(1)
            if LA13_0 == 117:
                alt13 = 1
            if alt13 == 1:
                pass
                self.match('uary')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mFEBRUARY(self):
        try:
            _type = FEBRUARY
            _channel = DEFAULT_CHANNEL
            pass
            self.match('feb')
            alt14 = 2
            LA14_0 = self.input.LA(1)
            if LA14_0 == 114:
                alt14 = 1
            if alt14 == 1:
                pass
                self.match('ruary')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mMARCH(self):
        try:
            _type = MARCH
            _channel = DEFAULT_CHANNEL
            pass
            self.match('mar')
            alt15 = 2
            LA15_0 = self.input.LA(1)
            if LA15_0 == 99:
                alt15 = 1
            if alt15 == 1:
                pass
                self.match('ch')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mAPRIL(self):
        try:
            _type = APRIL
            _channel = DEFAULT_CHANNEL
            pass
            self.match('apr')
            alt16 = 2
            LA16_0 = self.input.LA(1)
            if LA16_0 == 105:
                alt16 = 1
            if alt16 == 1:
                pass
                self.match('il')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mMAY(self):
        try:
            _type = MAY
            _channel = DEFAULT_CHANNEL
            pass
            self.match('may')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mJUNE(self):
        try:
            _type = JUNE
            _channel = DEFAULT_CHANNEL
            pass
            self.match('jun')
            alt17 = 2
            LA17_0 = self.input.LA(1)
            if LA17_0 == 101:
                alt17 = 1
            if alt17 == 1:
                pass
                self.match(101)
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mJULY(self):
        try:
            _type = JULY
            _channel = DEFAULT_CHANNEL
            pass
            self.match('jul')
            alt18 = 2
            LA18_0 = self.input.LA(1)
            if LA18_0 == 121:
                alt18 = 1
            if alt18 == 1:
                pass
                self.match(121)
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mAUGUST(self):
        try:
            _type = AUGUST
            _channel = DEFAULT_CHANNEL
            pass
            self.match('aug')
            alt19 = 2
            LA19_0 = self.input.LA(1)
            if LA19_0 == 117:
                alt19 = 1
            if alt19 == 1:
                pass
                self.match('ust')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mSEPTEMBER(self):
        try:
            _type = SEPTEMBER
            _channel = DEFAULT_CHANNEL
            pass
            self.match('sep')
            alt20 = 2
            LA20_0 = self.input.LA(1)
            if LA20_0 == 116:
                alt20 = 1
            if alt20 == 1:
                pass
                self.match('tember')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mOCTOBER(self):
        try:
            _type = OCTOBER
            _channel = DEFAULT_CHANNEL
            pass
            self.match('oct')
            alt21 = 2
            LA21_0 = self.input.LA(1)
            if LA21_0 == 111:
                alt21 = 1
            if alt21 == 1:
                pass
                self.match('ober')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mNOVEMBER(self):
        try:
            _type = NOVEMBER
            _channel = DEFAULT_CHANNEL
            pass
            self.match('nov')
            alt22 = 2
            LA22_0 = self.input.LA(1)
            if LA22_0 == 101:
                alt22 = 1
            if alt22 == 1:
                pass
                self.match('ember')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mDECEMBER(self):
        try:
            _type = DECEMBER
            _channel = DEFAULT_CHANNEL
            pass
            self.match('dec')
            alt23 = 2
            LA23_0 = self.input.LA(1)
            if LA23_0 == 101:
                alt23 = 1
            if alt23 == 1:
                pass
                self.match('ember')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mMONTH(self):
        try:
            _type = MONTH
            _channel = DEFAULT_CHANNEL
            pass
            pass
            self.match('month')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mQUARTER(self):
        try:
            _type = QUARTER
            _channel = DEFAULT_CHANNEL
            pass
            pass
            self.match('quarter')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mEVERY(self):
        try:
            _type = EVERY
            _channel = DEFAULT_CHANNEL
            pass
            pass
            self.match('every')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mHOURS(self):
        try:
            _type = HOURS
            _channel = DEFAULT_CHANNEL
            pass
            pass
            self.match('hours')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mMINUTES(self):
        try:
            _type = MINUTES
            _channel = DEFAULT_CHANNEL
            pass
            alt24 = 2
            LA24_0 = self.input.LA(1)
            if LA24_0 == 109:
                LA24_1 = self.input.LA(2)
                if LA24_1 == 105:
                    LA24_2 = self.input.LA(3)
                    if LA24_2 == 110:
                        LA24_3 = self.input.LA(4)
                        if LA24_3 == 115:
                            alt24 = 1
                        elif LA24_3 == 117:
                            alt24 = 2
                        else:
                            if self._state.backtracking > 0:
                                raise BacktrackingFailed
                            nvae = NoViableAltException('', 24, 3, self.input)
                            raise nvae
                    else:
                        if self._state.backtracking > 0:
                            raise BacktrackingFailed
                        nvae = NoViableAltException('', 24, 2, self.input)
                        raise nvae
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed
                    nvae = NoViableAltException('', 24, 1, self.input)
                    raise nvae
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed
                nvae = NoViableAltException('', 24, 0, self.input)
                raise nvae
            if alt24 == 1:
                pass
                self.match('mins')
            elif alt24 == 2:
                pass
                self.match('minutes')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mCOMMA(self):
        try:
            _type = COMMA
            _channel = DEFAULT_CHANNEL
            pass
            pass
            self.match(44)
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mOF(self):
        try:
            _type = OF
            _channel = DEFAULT_CHANNEL
            pass
            pass
            self.match('of')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mFROM(self):
        try:
            _type = FROM
            _channel = DEFAULT_CHANNEL
            pass
            pass
            self.match('from')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mTO(self):
        try:
            _type = TO
            _channel = DEFAULT_CHANNEL
            pass
            pass
            self.match('to')
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mWS(self):
        try:
            _type = WS
            _channel = DEFAULT_CHANNEL
            pass
            if 9 <= self.input.LA(1) <= 10 or self.input.LA(1) == 13 or self.input.LA(1) == 32:
                self.input.consume()
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed
                mse = MismatchedSetException(None, self.input)
                self.recover(mse)
                raise mse
            if self._state.backtracking == 0:
                _channel = HIDDEN
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mDIGIT(self):
        try:
            _type = DIGIT
            _channel = DEFAULT_CHANNEL
            pass
            pass
            self.matchRange(48, 57)
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mDIGITS(self):
        try:
            _type = DIGITS
            _channel = DEFAULT_CHANNEL
            pass
            alt25 = 4
            LA25_0 = self.input.LA(1)
            if 48 <= LA25_0 <= 57:
                LA25_1 = self.input.LA(2)
                if 48 <= LA25_1 <= 57:
                    LA25_2 = self.input.LA(3)
                    if 48 <= LA25_2 <= 57:
                        LA25_4 = self.input.LA(4)
                        if 48 <= LA25_4 <= 57:
                            LA25_6 = self.input.LA(5)
                            if 48 <= LA25_6 <= 57 and self.synpred1_Groc():
                                alt25 = 1
                            else:
                                alt25 = 2
                        else:
                            alt25 = 3
                    else:
                        alt25 = 4
                else:
                    if self._state.backtracking > 0:
                        raise BacktrackingFailed
                    nvae = NoViableAltException('', 25, 1, self.input)
                    raise nvae
            else:
                if self._state.backtracking > 0:
                    raise BacktrackingFailed
                nvae = NoViableAltException('', 25, 0, self.input)
                raise nvae
            if alt25 == 1:
                pass
                pass
                self.mDIGIT()
                self.mDIGIT()
                self.mDIGIT()
                self.mDIGIT()
                self.mDIGIT()
            elif alt25 == 2:
                pass
                pass
                self.mDIGIT()
                self.mDIGIT()
                self.mDIGIT()
                self.mDIGIT()
            elif alt25 == 3:
                pass
                pass
                self.mDIGIT()
                self.mDIGIT()
                self.mDIGIT()
            elif alt25 == 4:
                pass
                pass
                self.mDIGIT()
                self.mDIGIT()
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mUNKNOWN_TOKEN(self):
        try:
            _type = UNKNOWN_TOKEN
            _channel = DEFAULT_CHANNEL
            pass
            pass
            self.matchAny()
            self._state.type = _type
            self._state.channel = _channel
        finally:
            pass

    def mTokens(self):
        alt26 = 42
        alt26 = self.dfa26.predict(self.input)
        if alt26 == 1:
            pass
            self.mTIME()
        elif alt26 == 2:
            pass
            self.mTWO_DIGIT_HOUR_TIME()
        elif alt26 == 3:
            pass
            self.mSYNCHRONIZED()
        elif alt26 == 4:
            pass
            self.mFIRST()
        elif alt26 == 5:
            pass
            self.mSECOND()
        elif alt26 == 6:
            pass
            self.mTHIRD()
        elif alt26 == 7:
            pass
            self.mFOURTH()
        elif alt26 == 8:
            pass
            self.mFIFTH()
        elif alt26 == 9:
            pass
            self.mFOURTH_OR_FIFTH()
        elif alt26 == 10:
            pass
            self.mDAY()
        elif alt26 == 11:
            pass
            self.mMONDAY()
        elif alt26 == 12:
            pass
            self.mTUESDAY()
        elif alt26 == 13:
            pass
            self.mWEDNESDAY()
        elif alt26 == 14:
            pass
            self.mTHURSDAY()
        elif alt26 == 15:
            pass
            self.mFRIDAY()
        elif alt26 == 16:
            pass
            self.mSATURDAY()
        elif alt26 == 17:
            pass
            self.mSUNDAY()
        elif alt26 == 18:
            pass
            self.mJANUARY()
        elif alt26 == 19:
            pass
            self.mFEBRUARY()
        elif alt26 == 20:
            pass
            self.mMARCH()
        elif alt26 == 21:
            pass
            self.mAPRIL()
        elif alt26 == 22:
            pass
            self.mMAY()
        elif alt26 == 23:
            pass
            self.mJUNE()
        elif alt26 == 24:
            pass
            self.mJULY()
        elif alt26 == 25:
            pass
            self.mAUGUST()
        elif alt26 == 26:
            pass
            self.mSEPTEMBER()
        elif alt26 == 27:
            pass
            self.mOCTOBER()
        elif alt26 == 28:
            pass
            self.mNOVEMBER()
        elif alt26 == 29:
            pass
            self.mDECEMBER()
        elif alt26 == 30:
            pass
            self.mMONTH()
        elif alt26 == 31:
            pass
            self.mQUARTER()
        elif alt26 == 32:
            pass
            self.mEVERY()
        elif alt26 == 33:
            pass
            self.mHOURS()
        elif alt26 == 34:
            pass
            self.mMINUTES()
        elif alt26 == 35:
            pass
            self.mCOMMA()
        elif alt26 == 36:
            pass
            self.mOF()
        elif alt26 == 37:
            pass
            self.mFROM()
        elif alt26 == 38:
            pass
            self.mTO()
        elif alt26 == 39:
            pass
            self.mWS()
        elif alt26 == 40:
            pass
            self.mDIGIT()
        elif alt26 == 41:
            pass
            self.mDIGITS()
        elif alt26 == 42:
            pass
            self.mUNKNOWN_TOKEN()

    def synpred1_Groc_fragment(self):
        pass
        self.mDIGIT()
        self.mDIGIT()
        self.mDIGIT()
        self.mDIGIT()
        self.mDIGIT()

    def synpred2_Groc_fragment(self):
        pass
        self.mDIGIT()
        self.mDIGIT()
        self.mDIGIT()
        self.mDIGIT()

    def synpred2_Groc(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred2_Groc_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success

    def synpred1_Groc(self):
        self._state.backtracking += 1
        start = self.input.mark()
        try:
            self.synpred1_Groc_fragment()
        except BacktrackingFailed:
            success = False
        else:
            success = True
        self.input.rewind(start)
        self._state.backtracking -= 1
        return success
    DFA26_eot = DFA.unpack(u"\x01\uffff\x04\x18\x02\x17\x01\x18\x01\x17\x02\x18\n\x17\x04\uffff\x01\x1f\x02\uffff\x02\x1f'\uffff\x01K\x06\uffff")
    DFA26_eof = DFA.unpack(u'L\uffff')
    DFA26_min = DFA.unpack(u'\x01\x00\x040\x01a\x01e\x010\x01h\x020\x02a\x01e\x01a\x01p\x01c\x01o\x01u\x01v\x01o\x04\uffff\x01:\x02\uffff\x02:\x04\uffff\x01c\x02\uffff\x01f\x01\uffff\x01i\x02\uffff\x01i\x05\uffff\x01n\x01r\x03\uffff\x01l\x0f\uffff\x01t\x06\uffff')
    DFA26_max = DFA.unpack(u'\x01\uffff\x01:\x01s\x01n\x01r\x01y\x01r\x01t\x01u\x01t\x01:\x01e\x01o\x01e\x02u\x01f\x01o\x01u\x01v\x01o\x04\uffff\x01:\x02\uffff\x02:\x04\uffff\x01p\x02\uffff\x01r\x01\uffff\x01o\x02\uffff\x01u\x05\uffff\x01n\x01y\x03\uffff\x01n\x0f\uffff\x01t\x06\uffff')
    DFA26_accept = DFA.unpack(u'\x15\uffff\x01#\x01\'\x01*\x01(\x01\uffff\x01\x01\x01\x04\x02\uffff\x01\x05\x01)\x01\x06\x01\x03\x01\uffff\x01\x10\x01\x11\x01\uffff\x01\t\x01\uffff\x01\x13\x01\x07\x01\uffff\x01\x0c\x01&\x01\x08\x01\n\x01\x1d\x02\uffff\x01"\x01\r\x01\x12\x01\uffff\x01\x15\x01\x19\x01\x1b\x01$\x01\x1c\x01\x1f\x01 \x01!\x01#\x01\'\x01\x02\x01\x1a\x01\x0f\x01%\x01\x0e\x01\uffff\x01\x14\x01\x16\x01\x17\x01\x18\x01\x1e\x01\x0b')
    DFA26_special = DFA.unpack(u'\x01\x00K\uffff')
    DFA26_transition = [DFA.unpack(u"\t\x17\x02\x16\x02\x17\x01\x16\x12\x17\x01\x16\x0b\x17\x01\x15\x03\x17\x01\x01\x01\x02\x01\x03\x01\x04\x01\x07\x01\t\x04\n'\x17\x01\x0f\x02\x17\x01\x0b\x01\x13\x01\x06\x01\x17\x01\x14\x01\x17\x01\x0e\x02\x17\x01\x0c\x01\x11\x01\x10\x01\x17\x01\x12\x01\x17\x01\x05\x01\x08\x02\x17\x01\rﾈ\x17"), DFA.unpack(u'\n\x19\x01\x1a'), DFA.unpack(u'\n\x1c\x01\x1a8\uffff\x01\x1b'), DFA.unpack(u'\x04\x1d\x06\x1f\x01\x1a3\uffff\x01\x1e'), DFA.unpack(u'\n\x1f\x01\x1a7\uffff\x01 '), DFA.unpack(u'\x01#\x03\uffff\x01"\x0f\uffff\x01$\x03\uffff\x01!'), DFA.unpack(u"\x01(\x03\uffff\x01%\x05\uffff\x01&\x02\uffff\x01'"), DFA.unpack(u'\n\x1f\x01\x1a9\uffff\x01)'), DFA.unpack(u'\x01*\x06\uffff\x01,\x05\uffff\x01+'), DFA.unpack(u'\n\x1f\x01\x1a9\uffff\x01-'), DFA.unpack(u'\n\x1f\x01\x1a'), DFA.unpack(u'\x01.\x03\uffff\x01/'), DFA.unpack(u'\x011\x07\uffff\x012\x05\uffff\x010'), DFA.unpack(u'\x013'), DFA.unpack(u'\x014\x13\uffff\x015'), DFA.unpack(u'\x016\x04\uffff\x017'), DFA.unpack(u'\x018\x02\uffff\x019'), DFA.unpack(u'\x01:'), DFA.unpack(u'\x01;'), DFA.unpack(u'\x01<'), DFA.unpack(u'\x01='), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u'\x01@'), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u'\x01@'), DFA.unpack(u'\x01@'), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u'\x01\x1e\x0c\uffff\x01A'), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u'\x01&\x0b\uffff\x01\x1b'), DFA.unpack(u''), DFA.unpack(u'\x01B\x05\uffff\x01C'), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u'\x01 \x0b\uffff\x01D'), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u'\x01E'), DFA.unpack(u'\x01F\x06\uffff\x01G'), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u'\x01I\x01\uffff\x01H'), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u'\x01J'), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u''), DFA.unpack(u'')]

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