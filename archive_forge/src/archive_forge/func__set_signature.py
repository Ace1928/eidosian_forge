from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _set_signature(mock, original, instance=False):
    if not _callable(original):
        return
    skipfirst = isinstance(original, ClassTypes)
    result = _get_signature_object(original, instance, skipfirst)
    if result is None:
        return
    func, sig = result

    def checksig(*args, **kwargs):
        sig.bind(*args, **kwargs)
    _copy_func_details(func, checksig)
    name = original.__name__
    if not _isidentifier(name):
        name = 'funcopy'
    context = {'_checksig_': checksig, 'mock': mock}
    src = 'def %s(*args, **kwargs):\n    _checksig_(*args, **kwargs)\n    return mock(*args, **kwargs)' % name
    six.exec_(src, context)
    funcopy = context[name]
    _setup_func(funcopy, mock)
    return funcopy