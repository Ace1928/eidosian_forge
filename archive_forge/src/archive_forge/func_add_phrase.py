from parlai.agents.transformer.transformer import TransformerClassifierAgent
from parlai.core.agents import create_agent, create_agent_from_shared
from parlai.tasks.dialogue_safety.agents import OK_CLASS, NOT_OK_CLASS
from parlai.utils.typing import TShared
import parlai.utils.logging as logging
import os
def add_phrase(self, phrase):
    """
        Add a single phrase to the filter.
        """
    toks = self.tokenize(phrase)
    curr = self.offensive_trie
    for t in toks:
        if t not in curr:
            curr[t] = {}
        curr = curr[t]
    curr[self.END] = True
    self.max_len = max(self.max_len, len(toks))