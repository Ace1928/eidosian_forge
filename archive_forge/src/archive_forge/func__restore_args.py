import base64
import inspect
import builtins
@staticmethod
def _restore_args(dict_):

    def restore(k):
        if k in _RESERVED_KEYWORD:
            return k + '_'
        return k
    return _mapdict_key(restore, dict_)