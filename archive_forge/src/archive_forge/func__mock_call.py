from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _mock_call(_mock_self, *args, **kwargs):
    self = _mock_self
    self.called = True
    self.call_count += 1
    _new_name = self._mock_new_name
    _new_parent = self._mock_new_parent
    _call = _Call((args, kwargs), two=True)
    self.call_args = _call
    self.call_args_list.append(_call)
    self.mock_calls.append(_Call(('', args, kwargs)))
    seen = set()
    skip_next_dot = _new_name == '()'
    do_method_calls = self._mock_parent is not None
    name = self._mock_name
    while _new_parent is not None:
        this_mock_call = _Call((_new_name, args, kwargs))
        if _new_parent._mock_new_name:
            dot = '.'
            if skip_next_dot:
                dot = ''
            skip_next_dot = False
            if _new_parent._mock_new_name == '()':
                skip_next_dot = True
            _new_name = _new_parent._mock_new_name + dot + _new_name
        if do_method_calls:
            if _new_name == name:
                this_method_call = this_mock_call
            else:
                this_method_call = _Call((name, args, kwargs))
            _new_parent.method_calls.append(this_method_call)
            do_method_calls = _new_parent._mock_parent is not None
            if do_method_calls:
                name = _new_parent._mock_name + '.' + name
        _new_parent.mock_calls.append(this_mock_call)
        _new_parent = _new_parent._mock_new_parent
        _new_parent_id = id(_new_parent)
        if _new_parent_id in seen:
            break
        seen.add(_new_parent_id)
    ret_val = DEFAULT
    effect = self.side_effect
    if effect is not None:
        if _is_exception(effect):
            raise effect
        if not _callable(effect):
            result = next(effect)
            if _is_exception(result):
                raise result
            if result is DEFAULT:
                result = self.return_value
            return result
        ret_val = effect(*args, **kwargs)
    if self._mock_wraps is not None and self._mock_return_value is DEFAULT:
        return self._mock_wraps(*args, **kwargs)
    if ret_val is DEFAULT:
        ret_val = self.return_value
    return ret_val