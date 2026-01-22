from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.PredictionContext import PredictionContext, SingletonPredictionContext, PredictionContextFromRuleContext
from antlr4.RuleContext import RuleContext
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import ATNConfig
from antlr4.atn.ATNState import ATNState, RuleStopState
from antlr4.atn.Transition import WildcardTransition, NotSetTransition, AbstractPredicateTransition, RuleTransition
def _LOOK(self, s: ATNState, stopState: ATNState, ctx: PredictionContext, look: IntervalSet, lookBusy: set, calledRuleStack: set, seeThruPreds: bool, addEOF: bool):
    c = ATNConfig(s, 0, ctx)
    if c in lookBusy:
        return
    lookBusy.add(c)
    if s == stopState:
        if ctx is None:
            look.addOne(Token.EPSILON)
            return
        elif ctx.isEmpty() and addEOF:
            look.addOne(Token.EOF)
            return
    if isinstance(s, RuleStopState):
        if ctx is None:
            look.addOne(Token.EPSILON)
            return
        elif ctx.isEmpty() and addEOF:
            look.addOne(Token.EOF)
            return
        if ctx != PredictionContext.EMPTY:
            removed = s.ruleIndex in calledRuleStack
            try:
                calledRuleStack.discard(s.ruleIndex)
                for i in range(0, len(ctx)):
                    returnState = self.atn.states[ctx.getReturnState(i)]
                    self._LOOK(returnState, stopState, ctx.getParent(i), look, lookBusy, calledRuleStack, seeThruPreds, addEOF)
            finally:
                if removed:
                    calledRuleStack.add(s.ruleIndex)
            return
    for t in s.transitions:
        if type(t) == RuleTransition:
            if t.target.ruleIndex in calledRuleStack:
                continue
            newContext = SingletonPredictionContext.create(ctx, t.followState.stateNumber)
            try:
                calledRuleStack.add(t.target.ruleIndex)
                self._LOOK(t.target, stopState, newContext, look, lookBusy, calledRuleStack, seeThruPreds, addEOF)
            finally:
                calledRuleStack.remove(t.target.ruleIndex)
        elif isinstance(t, AbstractPredicateTransition):
            if seeThruPreds:
                self._LOOK(t.target, stopState, ctx, look, lookBusy, calledRuleStack, seeThruPreds, addEOF)
            else:
                look.addOne(self.HIT_PRED)
        elif t.isEpsilon:
            self._LOOK(t.target, stopState, ctx, look, lookBusy, calledRuleStack, seeThruPreds, addEOF)
        elif type(t) == WildcardTransition:
            look.addRange(range(Token.MIN_USER_TOKEN_TYPE, self.atn.maxTokenType + 1))
        else:
            set_ = t.label
            if set_ is not None:
                if isinstance(t, NotSetTransition):
                    set_ = set_.complement(Token.MIN_USER_TOKEN_TYPE, self.atn.maxTokenType)
                look.addSet(set_)