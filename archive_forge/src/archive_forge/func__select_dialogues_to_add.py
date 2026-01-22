import os
from typing import Any, List
import numpy as np
from parlai.core.teachers import FixedDialogTeacher
from .build import build
def _select_dialogues_to_add(self, experiencer_text_dialogue: List[List[Any]], responder_text_dialogue: List[List[Any]]) -> List[List[List[Any]]]:
    """
        Return conversation halves to add to self.data.

        Given lists corresponding to the conversation turns from both sides of the
        conversation, return only the list(s) that will be added to self.data.
        Optionally filter by side of the conversation or by whether the conversation
        contains any political language.
        """
    if self.remove_political_convos and any([turn[9] for turn in experiencer_text_dialogue + responder_text_dialogue]):
        return []
    else:
        selected_dialogues = []
        if len(experiencer_text_dialogue) > 0:
            selected_dialogues.append(experiencer_text_dialogue)
        if len(responder_text_dialogue) > 0 and (not self.experiencer_side_only):
            selected_dialogues.append(responder_text_dialogue)
        return selected_dialogues