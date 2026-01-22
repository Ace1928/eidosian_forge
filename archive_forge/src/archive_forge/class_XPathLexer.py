from antlr4 import *
from io import StringIO
import sys
class XPathLexer(Lexer):
    atn = ATNDeserializer().deserialize(serializedATN())
    decisionsToDFA = [DFA(ds, i) for i, ds in enumerate(atn.decisionToState)]
    TOKEN_REF = 1
    RULE_REF = 2
    ANYWHERE = 3
    ROOT = 4
    WILDCARD = 5
    BANG = 6
    ID = 7
    STRING = 8
    channelNames = [u'DEFAULT_TOKEN_CHANNEL', u'HIDDEN']
    modeNames = ['DEFAULT_MODE']
    literalNames = ['<INVALID>', "'//'", "'/'", "'*'", "'!'"]
    symbolicNames = ['<INVALID>', 'TOKEN_REF', 'RULE_REF', 'ANYWHERE', 'ROOT', 'WILDCARD', 'BANG', 'ID', 'STRING']
    ruleNames = ['ANYWHERE', 'ROOT', 'WILDCARD', 'BANG', 'ID', 'NameChar', 'NameStartChar', 'STRING']
    grammarFileName = 'XPathLexer.g4'

    def __init__(self, input=None, output: TextIO=sys.stdout):
        super().__init__(input, output)
        self.checkVersion('4.9.3')
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None

    def action(self, localctx: RuleContext, ruleIndex: int, actionIndex: int):
        if self._actions is None:
            actions = dict()
            actions[4] = self.ID_action
            self._actions = actions
        action = self._actions.get(ruleIndex, None)
        if action is not None:
            action(localctx, actionIndex)
        else:
            raise Exception('No registered action for:' + str(ruleIndex))

    def ID_action(self, localctx: RuleContext, actionIndex: int):
        if actionIndex == 0:
            char = self.text[0]
            if char.isupper():
                self.type = XPathLexer.TOKEN_REF
            else:
                self.type = XPathLexer.RULE_REF