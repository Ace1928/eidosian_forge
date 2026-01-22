import bisect
import textwrap
from collections import defaultdict
from nltk.tag import BrillTagger, untag
def _update_rule_applies(self, rule, sentnum, wordnum, train_sents):
    """
        Update the rule data tables to reflect the fact that
        *rule* applies at the position *(sentnum, wordnum)*.
        """
    pos = (sentnum, wordnum)
    if pos in self._positions_by_rule[rule]:
        return
    correct_tag = train_sents[sentnum][wordnum][1]
    if rule.replacement_tag == correct_tag:
        self._positions_by_rule[rule][pos] = 1
    elif rule.original_tag == correct_tag:
        self._positions_by_rule[rule][pos] = -1
    else:
        self._positions_by_rule[rule][pos] = 0
    self._rules_by_position[pos].add(rule)
    old_score = self._rule_scores[rule]
    self._rule_scores[rule] += self._positions_by_rule[rule][pos]
    self._rules_by_score[old_score].discard(rule)
    self._rules_by_score[self._rule_scores[rule]].add(rule)