from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_dataset_ops
def _parse_args(self, *args, **kwargs):
    """Parses arguments according to the same rules as the `range()` builtin."""
    if len(args) == 1:
        self._start = self._build_tensor(0, 'start')
        self._stop = self._build_tensor(args[0], 'stop')
        self._step = self._build_tensor(1, 'step')
    elif len(args) == 2:
        self._start = self._build_tensor(args[0], 'start')
        self._stop = self._build_tensor(args[1], 'stop')
        self._step = self._build_tensor(1, 'step')
    elif len(args) == 3:
        self._start = self._build_tensor(args[0], 'start')
        self._stop = self._build_tensor(args[1], 'stop')
        self._step = self._build_tensor(args[2], 'step')
    else:
        raise ValueError(f'Invalid `args`. The length of `args` should be between 1 and 3 but was {len(args)}.')
    if 'output_type' in kwargs:
        self._output_type = kwargs['output_type']
    else:
        self._output_type = dtypes.int64
    self._name = kwargs['name'] if 'name' in kwargs else None