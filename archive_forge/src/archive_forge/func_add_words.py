from parlai.agents.transformer.transformer import TransformerClassifierAgent
from parlai.core.agents import create_agent, create_agent_from_shared
from parlai.tasks.dialogue_safety.agents import OK_CLASS, NOT_OK_CLASS
from parlai.utils.typing import TShared
import parlai.utils.logging as logging
import os
def add_words(self, phrase_list):
    """
        Add list of custom phrases to the filter.
        """
    for phrase in phrase_list:
        self.add_phrase(phrase)