from antlr4.atn.ATNState import StarLoopEntryState
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.atn.ATNState import DecisionState
from antlr4.dfa.DFAState import DFAState
from antlr4.error.Errors import IllegalStateException
def toLexerString(self):
    if self.s0 is None:
        return ''
    from antlr4.dfa.DFASerializer import LexerDFASerializer
    serializer = LexerDFASerializer(self)
    return str(serializer)