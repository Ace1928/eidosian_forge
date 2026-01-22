import bisect
import textwrap
from collections import defaultdict
from nltk.tag import BrillTagger, untag
def _trace_rule(self, rule):
    assert self._rule_scores[rule] == sum(self._positions_by_rule[rule].values())
    changes = self._positions_by_rule[rule].values()
    num_fixed = len([c for c in changes if c == 1])
    num_broken = len([c for c in changes if c == -1])
    num_other = len([c for c in changes if c == 0])
    score = self._rule_scores[rule]
    rulestr = rule.format(self._ruleformat)
    if self._trace > 2:
        print('{:4d}{:4d}{:4d}{:4d}  |'.format(score, num_fixed, num_broken, num_other), end=' ')
        print(textwrap.fill(rulestr, initial_indent=' ' * 20, width=79, subsequent_indent=' ' * 18 + '|   ').strip())
    else:
        print(rulestr)