import inspect
import functools
from enum import Enum
import torch.autograd
@functools.wraps(func)
def wrap_generator(*args, **kwargs):
    gen = func(*args, **kwargs)
    datapipe = args[0]
    if datapipe._fast_forward_iterator:
        it = datapipe._fast_forward_iterator
        datapipe._fast_forward_iterator = None
        datapipe._snapshot_state = _SnapshotState.Iterating
        while True:
            try:
                yield next(it)
            except StopIteration:
                return
    iterator_id = _set_datapipe_valid_iterator_id(datapipe)
    _profiler_enabled = torch.autograd._profiler_enabled()
    try:
        if _profiler_enabled:
            with profiler_record_fn_context(datapipe):
                response = gen.send(None)
        else:
            response = gen.send(None)
        while True:
            datapipe._number_of_samples_yielded += 1
            request = (yield response)
            if _profiler_enabled:
                with profiler_record_fn_context(datapipe):
                    _check_iterator_valid(datapipe, iterator_id)
                    response = gen.send(request)
            else:
                _check_iterator_valid(datapipe, iterator_id)
                response = gen.send(request)
    except StopIteration as e:
        return
    except Exception as e:
        datapipe = args[0]
        msg = 'thrown by __iter__ of'
        single_iterator_msg = 'single iterator per IterDataPipe constraint'
        if hasattr(e.args, '__len__'):
            full_msg = f'{msg} {datapipe.__class__.__name__}({_generate_input_args_string(datapipe)})'
            if len(e.args) == 0 or not isinstance(e.args[0], str):
                e.args = (f'\nThis exception is {full_msg}',)
            elif msg not in e.args[0] and single_iterator_msg not in e.args[0]:
                e.args = (e.args[0] + f'\nThis exception is {full_msg}',) + e.args[1:]
        raise