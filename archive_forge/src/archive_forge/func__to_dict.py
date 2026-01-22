import inspect
import types
from wandb.errors import UsageError
from .lib import config_util
def _to_dict(params):
    if isinstance(params, dict):
        return params
    meta = inspect.getmodule(params)
    if meta:
        is_tf_flags_module = isinstance(params, types.ModuleType) and meta.__name__ == 'tensorflow.python.platform.flags'
        if is_tf_flags_module or meta.__name__ == 'absl.flags':
            params = params.FLAGS
            meta = inspect.getmodule(params)
    if meta and meta.__name__ == 'absl.flags._flagvalues':
        params = {name: params[name].value for name in dir(params)}
    elif not hasattr(params, '__dict__'):
        raise TypeError('config must be a dict or have a __dict__ attribute.')
    elif '__flags' in vars(params):
        if not '__parsed' not in vars(params):
            params._parse_flags()
        params = vars(params)['__flags']
    else:
        params = vars(params)
    return params