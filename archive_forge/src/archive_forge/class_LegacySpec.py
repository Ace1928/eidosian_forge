import functools
import re
import warnings
class LegacySpec(SimpleSpec):

    def __init__(self, *expressions):
        warnings.warn('The Spec() class will be removed in 3.1; use SimpleSpec() instead.', PendingDeprecationWarning, stacklevel=2)
        if len(expressions) > 1:
            warnings.warn("Passing 2+ arguments to SimpleSpec will be removed in 3.0; concatenate them with ',' instead.", DeprecationWarning, stacklevel=2)
        expression = ','.join(expressions)
        super(LegacySpec, self).__init__(expression)

    @property
    def specs(self):
        return list(self)

    def __iter__(self):
        warnings.warn('Iterating over the components of a SimpleSpec object will be removed in 3.0.', DeprecationWarning, stacklevel=2)
        try:
            clauses = list(self.clause)
        except TypeError:
            clauses = [self.clause]
        for clause in clauses:
            yield SpecItem.from_matcher(clause)