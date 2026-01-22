import itertools as it
from abc import ABCMeta, abstractmethod
from nltk.tbl.feature import Feature
from nltk.tbl.rule import Rule
def applicable_rules(self, tokens, index, correct_tag):
    if tokens[index][1] == correct_tag:
        return []
    applicable_conditions = self._applicable_conditions(tokens, index)
    xs = list(it.product(*applicable_conditions))
    return [Rule(self.id, tokens[index][1], correct_tag, tuple(x)) for x in xs]