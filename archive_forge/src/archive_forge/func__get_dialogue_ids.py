from typing import Dict, List, Set, Any
import json
import os
import queue
import random
import time
from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_manager import StaticMTurkManager
from parlai.mturk.core.worlds import StaticMTurkTaskWorld
from parlai.utils.misc import warn_once
def _get_dialogue_ids(self, task: Dict[str, Any]) -> List[int]:
    """
        Return the ids for the dialogues corresponding to a given task.

        :return dialogue_ids:
            A list of two ids which correspond to the id for each conversation
        """
    return task['pairing_dict']['dialogue_ids']