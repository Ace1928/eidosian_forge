from tensorflow.core.protobuf import cluster_pb2
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.training import server_lib
def worker_count(cluster_spec, task_type):
    """Returns the number of workers in the cluster."""
    _validate_cluster_spec(cluster_spec, task_type, task_id=0)
    cluster_spec = normalize_cluster_spec(cluster_spec).as_dict()
    if task_type not in ['chief', 'worker', 'evaluator']:
        raise ValueError('Unexpected `task_type` %r' % task_type)
    if task_type == 'evaluator':
        return len(cluster_spec['evaluator'])
    else:
        return len(cluster_spec.get('chief', [])) + len(cluster_spec.get('worker', []))