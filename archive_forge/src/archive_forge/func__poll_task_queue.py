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
def _poll_task_queue(self, worker_id: str, task_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
        Poll task queue for tasks for a worker.

        :param worker_id:
            id for worker

        :param task_data:
            list of potential tasks already for worker

        :return task_data:
            a list of tasks for a worker to complete
        """
    worker_data = self._get_worker_data(worker_id)
    num_attempts = 0
    while not self.task_queue.empty() and num_attempts < self.task_queue.qsize():
        try:
            next_task = self.task_queue.get()
        except queue.Empty:
            break
        num_attempts += 1
        pair_id = next_task['pair_id']
        dialogue_ids = self._get_dialogue_ids(next_task)
        if pair_id not in worker_data['tasks_completed'] and all((d_id not in worker_data['conversations_seen'] for d_id in dialogue_ids)):
            worker_data['tasks_completed'].append(pair_id)
            worker_data['conversations_seen'].extend(dialogue_ids)
            task_data.append(next_task)
            if len(task_data) == self.opt['subtasks_per_hit']:
                return task_data
        else:
            self.task_queue.put(next_task)
    return task_data