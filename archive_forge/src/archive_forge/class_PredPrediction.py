from io import StringIO
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.atn.SemanticContext import SemanticContext
class PredPrediction(object):
    __slots__ = ('alt', 'pred')

    def __init__(self, pred: SemanticContext, alt: int):
        self.alt = alt
        self.pred = pred

    def __str__(self):
        return '(' + str(self.pred) + ', ' + str(self.alt) + ')'