import multiprocessing
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
def _snapshot(input_dataset, path, compression='AUTO', reader_func=None, shard_func=None, name=None):
    """See `Dataset.snapshot()` for details."""
    project_func = None
    if shard_func is None:
        input_dataset = input_dataset.enumerate(name=name)
        local_shard_func = lambda index, _: index % multiprocessing.cpu_count()
        project_func = lambda _, elem: elem
    else:
        local_shard_func = shard_func
    dataset = _SnapshotDataset(input_dataset=input_dataset, path=path, compression=compression, reader_func=reader_func, shard_func=local_shard_func, name=name)
    if project_func is not None:
        dataset = dataset.map(project_func, name=name)
    return dataset