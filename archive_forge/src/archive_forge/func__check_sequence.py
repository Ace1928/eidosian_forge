from parlai.agents.transformer.transformer import TransformerClassifierAgent
from parlai.core.agents import create_agent, create_agent_from_shared
from parlai.tasks.dialogue_safety.agents import OK_CLASS, NOT_OK_CLASS
from parlai.utils.typing import TShared
import parlai.utils.logging as logging
import os
def _check_sequence(self, toks, idx, node):
    """
        Check if words from the sequence are in the trie.

        This checks phrases made from toks[i], toks[i:i+2] ... toks[i:i + self.max_len]
        """
    right = min(idx + self.max_len, len(toks))
    for i in range(idx, right):
        if toks[i] in node:
            node = node[toks[i]]
            if self.END in node:
                return ' '.join((toks[j] for j in range(idx, i + 1)))
        else:
            break
    return False