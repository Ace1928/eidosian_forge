from .task_list import task_list
from collections import defaultdict
def _id_to_task(t_id):
    if t_id[0] == '#':
        return ','.join((d['task'] for d in _id_to_task_data(t_id[1:])))
    else:
        return t_id