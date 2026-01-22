import bisect
import textwrap
from collections import defaultdict
from nltk.tag import BrillTagger, untag
def _best_rule(self, train_sents, test_sents, min_score, min_acc):
    """
        Find the next best rule.  This is done by repeatedly taking a
        rule with the highest score and stepping through the corpus to
        see where it applies.  When it makes an error (decreasing its
        score) it's bumped down, and we try a new rule with the
        highest score.  When we find a rule which has the highest
        score *and* which has been tested against the entire corpus, we
        can conclude that it's the next best rule.
        """
    for max_score in sorted(self._rules_by_score.keys(), reverse=True):
        if len(self._rules_by_score) == 0:
            return None
        if max_score < min_score or max_score <= 0:
            return None
        best_rules = list(self._rules_by_score[max_score])
        if self._deterministic:
            best_rules.sort(key=repr)
        for rule in best_rules:
            positions = self._tag_positions[rule.original_tag]
            unk = self._first_unknown_position.get(rule, (0, -1))
            start = bisect.bisect_left(positions, unk)
            for i in range(start, len(positions)):
                sentnum, wordnum = positions[i]
                if rule.applies(test_sents[sentnum], wordnum):
                    self._update_rule_applies(rule, sentnum, wordnum, train_sents)
                    if self._rule_scores[rule] < max_score:
                        self._first_unknown_position[rule] = (sentnum, wordnum + 1)
                        break
            if self._rule_scores[rule] == max_score:
                self._first_unknown_position[rule] = (len(train_sents) + 1, 0)
                if min_acc is None:
                    return rule
                else:
                    changes = self._positions_by_rule[rule].values()
                    num_fixed = len([c for c in changes if c == 1])
                    num_broken = len([c for c in changes if c == -1])
                    acc = num_fixed / (num_fixed + num_broken)
                    if acc >= min_acc:
                        return rule
        assert min_acc is not None or not self._rules_by_score[max_score]
        if not self._rules_by_score[max_score]:
            del self._rules_by_score[max_score]