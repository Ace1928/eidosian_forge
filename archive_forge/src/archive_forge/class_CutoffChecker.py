import math
import nltk.classify.util  # for accuracy & log_likelihood
from nltk.util import LazyMap
class CutoffChecker:
    """
    A helper class that implements cutoff checks based on number of
    iterations and log likelihood.

    Accuracy cutoffs are also implemented, but they're almost never
    a good idea to use.
    """

    def __init__(self, cutoffs):
        self.cutoffs = cutoffs.copy()
        if 'min_ll' in cutoffs:
            cutoffs['min_ll'] = -abs(cutoffs['min_ll'])
        if 'min_lldelta' in cutoffs:
            cutoffs['min_lldelta'] = abs(cutoffs['min_lldelta'])
        self.ll = None
        self.acc = None
        self.iter = 1

    def check(self, classifier, train_toks):
        cutoffs = self.cutoffs
        self.iter += 1
        if 'max_iter' in cutoffs and self.iter >= cutoffs['max_iter']:
            return True
        new_ll = nltk.classify.util.log_likelihood(classifier, train_toks)
        if math.isnan(new_ll):
            return True
        if 'min_ll' in cutoffs or 'min_lldelta' in cutoffs:
            if 'min_ll' in cutoffs and new_ll >= cutoffs['min_ll']:
                return True
            if 'min_lldelta' in cutoffs and self.ll and (new_ll - self.ll <= abs(cutoffs['min_lldelta'])):
                return True
            self.ll = new_ll
        if 'max_acc' in cutoffs or 'min_accdelta' in cutoffs:
            new_acc = nltk.classify.util.log_likelihood(classifier, train_toks)
            if 'max_acc' in cutoffs and new_acc >= cutoffs['max_acc']:
                return True
            if 'min_accdelta' in cutoffs and self.acc and (new_acc - self.acc <= abs(cutoffs['min_accdelta'])):
                return True
            self.acc = new_acc
            return False