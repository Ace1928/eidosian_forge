from .task_list import task_list
from collections import defaultdict
def _id_to_task_data(t_id):
    t_id = _preprocess(t_id)
    if t_id in tasks:
        return tasks[t_id]
    elif t_id in tags:
        return tags[t_id]
    else:
        raise RuntimeError('could not find tag/task id')