import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
class DrtGlueFormula(GlueFormula):

    def __init__(self, meaning, glue, indices=None):
        if not indices:
            indices = set()
        if isinstance(meaning, str):
            self.meaning = drt.DrtExpression.fromstring(meaning)
        elif isinstance(meaning, drt.DrtExpression):
            self.meaning = meaning
        else:
            raise RuntimeError('Meaning term neither string or expression: %s, %s' % (meaning, meaning.__class__))
        if isinstance(glue, str):
            self.glue = linearlogic.LinearLogicParser().parse(glue)
        elif isinstance(glue, linearlogic.Expression):
            self.glue = glue
        else:
            raise RuntimeError('Glue term neither string or expression: %s, %s' % (glue, glue.__class__))
        self.indices = indices

    def make_VariableExpression(self, name):
        return drt.DrtVariableExpression(name)

    def make_LambdaExpression(self, variable, term):
        return drt.DrtLambdaExpression(variable, term)