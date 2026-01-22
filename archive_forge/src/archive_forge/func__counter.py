from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
def _counter(start, step, dtype, name=None):
    with ops.name_scope('counter'):
        start = ops.convert_to_tensor(start, dtype=dtype, name='start')
        step = ops.convert_to_tensor(step, dtype=dtype, name='step')
        return dataset_ops.Dataset.from_tensors(0, name=name).repeat(None).scan(start, lambda state, _: (state + step, state))