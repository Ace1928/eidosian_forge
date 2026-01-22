from nltk.probability import ProbabilisticMixIn
from nltk.tree.parented import MultiParentedTree, ParentedTree
from nltk.tree.tree import Tree
class ImmutableProbabilisticTree(ImmutableTree, ProbabilisticMixIn):

    def __init__(self, node, children=None, **prob_kwargs):
        ImmutableTree.__init__(self, node, children)
        ProbabilisticMixIn.__init__(self, **prob_kwargs)
        self._hash = hash((self._label, tuple(self), self.prob()))

    def _frozen_class(self):
        return ImmutableProbabilisticTree

    def __repr__(self):
        return f'{Tree.__repr__(self)} [{self.prob()}]'

    def __str__(self):
        return f'{self.pformat(margin=60)} [{self.prob()}]'

    def copy(self, deep=False):
        if not deep:
            return type(self)(self._label, self, prob=self.prob())
        else:
            return type(self).convert(self)

    @classmethod
    def convert(cls, val):
        if isinstance(val, Tree):
            children = [cls.convert(child) for child in val]
            if isinstance(val, ProbabilisticMixIn):
                return cls(val._label, children, prob=val.prob())
            else:
                return cls(val._label, children, prob=1.0)
        else:
            return val