import logging
import re
from collections import namedtuple
from datetime import time
from urllib.parse import ParseResult, quote, urlparse, urlunparse
def _get_matching_rule_set(self, user_agent):
    """Return the rule set with highest matching score."""
    if not self._user_agents:
        return None
    if user_agent in self._matched_rule_set:
        return self._matched_rule_set[user_agent]
    score_rule_set_pairs = ((rs.applies_to(user_agent), rs) for rs in self._user_agents.values())
    match_score, matched_rule_set = max(score_rule_set_pairs, key=lambda p: p[0])
    if not match_score:
        self._matched_rule_set[user_agent] = None
        return None
    self._matched_rule_set[user_agent] = matched_rule_set
    return matched_rule_set