import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
def applyto(self, arg):
    """self = (\\x.(walk x), (subj -o f))
        arg  = (john        ,  subj)
        returns ((walk john),          f)
        """
    if self.indices & arg.indices:
        raise linearlogic.LinearLogicApplicationException(f"'{self}' applied to '{arg}'.  Indices are not disjoint.")
    else:
        return_indices = self.indices | arg.indices
    try:
        return_glue = linearlogic.ApplicationExpression(self.glue, arg.glue, arg.indices)
    except linearlogic.LinearLogicApplicationException as e:
        raise linearlogic.LinearLogicApplicationException(f"'{self.simplify()}' applied to '{arg.simplify()}'") from e
    arg_meaning_abstracted = arg.meaning
    if return_indices:
        for dep in self.glue.simplify().antecedent.dependencies[::-1]:
            arg_meaning_abstracted = self.make_LambdaExpression(Variable('v%s' % dep), arg_meaning_abstracted)
    return_meaning = self.meaning.applyto(arg_meaning_abstracted)
    return self.__class__(return_meaning, return_glue, return_indices)