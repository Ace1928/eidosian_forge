from io import StringIO
from antlr4.error.Errors import IllegalStateException
from antlr4.RuleContext import RuleContext
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNState import ATNState
class PredictionContextCache(object):

    def __init__(self):
        self.cache = dict()

    def add(self, ctx: PredictionContext):
        if ctx == PredictionContext.EMPTY:
            return PredictionContext.EMPTY
        existing = self.cache.get(ctx, None)
        if existing is not None:
            return existing
        self.cache[ctx] = ctx
        return ctx

    def get(self, ctx: PredictionContext):
        return self.cache.get(ctx, None)

    def __len__(self):
        return len(self.cache)