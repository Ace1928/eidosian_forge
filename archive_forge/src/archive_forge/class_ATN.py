from antlr4.IntervalSet import IntervalSet
from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.atn.ATNType import ATNType
from antlr4.atn.ATNState import ATNState, DecisionState
class ATN(object):
    __slots__ = ('grammarType', 'maxTokenType', 'states', 'decisionToState', 'ruleToStartState', 'ruleToStopState', 'modeNameToStartState', 'ruleToTokenType', 'lexerActions', 'modeToStartState')
    INVALID_ALT_NUMBER = 0

    def __init__(self, grammarType: ATNType, maxTokenType: int):
        self.grammarType = grammarType
        self.maxTokenType = maxTokenType
        self.states = []
        self.decisionToState = []
        self.ruleToStartState = []
        self.ruleToStopState = None
        self.modeNameToStartState = dict()
        self.ruleToTokenType = None
        self.lexerActions = None
        self.modeToStartState = []

    def nextTokensInContext(self, s: ATNState, ctx: RuleContext):
        from antlr4.LL1Analyzer import LL1Analyzer
        anal = LL1Analyzer(self)
        return anal.LOOK(s, ctx=ctx)

    def nextTokensNoContext(self, s: ATNState):
        if s.nextTokenWithinRule is not None:
            return s.nextTokenWithinRule
        s.nextTokenWithinRule = self.nextTokensInContext(s, None)
        s.nextTokenWithinRule.readonly = True
        return s.nextTokenWithinRule

    def nextTokens(self, s: ATNState, ctx: RuleContext=None):
        if ctx == None:
            return self.nextTokensNoContext(s)
        else:
            return self.nextTokensInContext(s, ctx)

    def addState(self, state: ATNState):
        if state is not None:
            state.atn = self
            state.stateNumber = len(self.states)
        self.states.append(state)

    def removeState(self, state: ATNState):
        self.states[state.stateNumber] = None

    def defineDecisionState(self, s: DecisionState):
        self.decisionToState.append(s)
        s.decision = len(self.decisionToState) - 1
        return s.decision

    def getDecisionState(self, decision: int):
        if len(self.decisionToState) == 0:
            return None
        else:
            return self.decisionToState[decision]

    def getExpectedTokens(self, stateNumber: int, ctx: RuleContext):
        if stateNumber < 0 or stateNumber >= len(self.states):
            raise Exception('Invalid state number.')
        s = self.states[stateNumber]
        following = self.nextTokens(s)
        if Token.EPSILON not in following:
            return following
        expected = IntervalSet()
        expected.addSet(following)
        expected.removeOne(Token.EPSILON)
        while ctx != None and ctx.invokingState >= 0 and (Token.EPSILON in following):
            invokingState = self.states[ctx.invokingState]
            rt = invokingState.transitions[0]
            following = self.nextTokens(rt.followState)
            expected.addSet(following)
            expected.removeOne(Token.EPSILON)
            ctx = ctx.parentCtx
        if Token.EPSILON in following:
            expected.addOne(Token.EOF)
        return expected