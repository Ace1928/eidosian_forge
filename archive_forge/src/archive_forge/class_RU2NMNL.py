import numpy as np
import numpy.lib.recfunctions as recf
from scipy import optimize
class RU2NMNL:
    """Nested Multinomial Logit with Random Utility 2 parameterization

    """

    def __init__(self, endog, exog, tree, paramsind):
        self.endog = endog
        self.datadict = exog
        self.tree = tree
        self.paramsind = paramsind
        self.branchsum = ''
        self.probs = {}

    def calc_prob(self, tree, keys=None):
        """walking a tree bottom-up based on dictionary
        """
        endog = self.endog
        datadict = self.datadict
        paramsind = self.paramsind
        branchsum = self.branchsum
        if isinstance(tree, tuple):
            name, subtree = tree
            print(name, datadict[name])
            print('subtree', subtree)
            keys = []
            if testxb:
                branchsum = datadict[name]
            else:
                branchsum = name
            for b in subtree:
                print(b)
                branchsum = branchsum + self.calc_prob(b, keys)
            print('branchsum', branchsum, keys)
            for k in keys:
                self.probs[k] = self.probs[k] + ['*' + name + '-prob']
        else:
            keys.append(tree)
            self.probs[tree] = [tree + '-prob' + '(%s)' % ', '.join(self.paramsind[tree])]
            if testxb:
                leavessum = sum((datadict[bi] for bi in tree))
                print('final branch with', tree, ''.join(tree), leavessum)
                return leavessum
            else:
                return ''.join(tree)
        print('working on branch', tree, branchsum)
        return branchsum