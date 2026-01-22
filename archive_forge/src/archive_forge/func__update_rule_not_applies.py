import bisect
import textwrap
from collections import defaultdict
from nltk.tag import BrillTagger, untag
def _update_rule_not_applies(self, rule, sentnum, wordnum):
    """
        Update the rule data tables to reflect the fact that *rule*
        does not apply at the position *(sentnum, wordnum)*.
        """
    pos = (sentnum, wordnum)
    old_score = self._rule_scores[rule]
    self._rule_scores[rule] -= self._positions_by_rule[rule][pos]
    self._rules_by_score[old_score].discard(rule)
    self._rules_by_score[self._rule_scores[rule]].add(rule)
    del self._positions_by_rule[rule][pos]
    self._rules_by_position[pos].remove(rule)