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
def _get_worker_data(self, worker_id: str) -> Dict[str, List]:
    """
        Return worker data if present, else a default dict.
        """
    onboarding_todo = list(range(len(self.onboarding_tasks)))
    random.shuffle(onboarding_todo)
    self.worker_data[worker_id] = self.worker_data.get(worker_id, {'tasks_completed': [], 'conversations_seen': [], 'onboarding_todo': onboarding_todo})
    return self.worker_data[worker_id]