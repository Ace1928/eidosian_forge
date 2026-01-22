import copy
import itertools
from pyomo.common import DeveloperError
from pyomo.common.collections import Sequence
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.global_set import UnindexedComponent_index
class _IndexedComponent_slice_iter(object):

    def __init__(self, component_slice, advance_iter=_advance_iter, iter_over_index=False, sort=False):
        self._slice = component_slice
        self.advance_iter = advance_iter
        self._iter_over_index = iter_over_index
        self._sort = SortComponents(sort)
        call_stack = self._slice._call_stack
        call_stack_len = self._slice._len
        self._iter_stack = [None] * call_stack_len
        if call_stack[0][0] == IndexedComponent_slice.slice_info:
            self._iter_stack[0] = _slice_generator(*call_stack[0][1], iter_over_index=self._iter_over_index, sort=self._sort)
        elif call_stack[0][0] == IndexedComponent_slice.set_item:
            assert call_stack_len == 1
            self._iter_stack[0] = _NotIterable
        else:
            raise DeveloperError('Unexpected call_stack flag encountered: %s' % call_stack[0][0])

    def __iter__(self):
        """This class implements the iterator API"""
        return self

    def next(self):
        """__next__() iterator for Py2 compatibility"""
        return self.__next__()

    def __next__(self):
        """Return the next element in the slice."""
        idx = len(self._iter_stack) - 1
        while True:
            while self._iter_stack[idx] is None:
                idx -= 1
            try:
                if self._iter_stack[idx] is _NotIterable:
                    _comp = self._slice._call_stack[0][1][0]
                else:
                    _comp = self.advance_iter(self._iter_stack[idx])
                    idx += 1
            except StopIteration:
                if not idx:
                    raise
                self._iter_stack[idx] = None
                idx -= 1
                continue
            while idx < self._slice._len:
                _call = self._slice._call_stack[idx]
                if _call[0] == IndexedComponent_slice.get_attribute:
                    try:
                        _comp = getattr(_comp, _call[1])
                    except AttributeError:
                        if self._slice.attribute_errors_generate_exceptions and (not self._iter_over_index):
                            raise
                        break
                elif _call[0] == IndexedComponent_slice.get_item:
                    try:
                        _comp = _comp.__getitem__(_call[1])
                    except LookupError:
                        if self._slice.key_errors_generate_exceptions and (not self._iter_over_index):
                            raise
                        break
                    if _comp.__class__ is IndexedComponent_slice:
                        assert _comp._len == 1
                        self._iter_stack[idx] = _slice_generator(*_comp._call_stack[0][1], iter_over_index=self._iter_over_index, sort=self._sort)
                        try:
                            _comp = self.advance_iter(self._iter_stack[idx])
                        except StopIteration:
                            self._iter_stack[idx] = None
                            break
                    else:
                        self._iter_stack[idx] = None
                elif _call[0] == IndexedComponent_slice.call:
                    try:
                        _comp = _comp(*_call[1], **_call[2])
                    except:
                        if self._slice.call_errors_generate_exceptions and (not self._iter_over_index):
                            raise
                        break
                elif _call[0] == IndexedComponent_slice.set_attribute:
                    assert idx == self._slice._len - 1
                    try:
                        _comp = setattr(_comp, _call[1], _call[2])
                    except AttributeError:
                        if self._slice.attribute_errors_generate_exceptions:
                            raise
                        break
                elif _call[0] == IndexedComponent_slice.set_item:
                    assert idx == self._slice._len - 1
                    if self._iter_stack[idx] is _NotIterable:
                        _iter = _slice_generator(*_call[1], iter_over_index=self._iter_over_index, sort=self._sort)
                        while True:
                            self.advance_iter(_iter)
                            self.advance_iter.check_complete()
                            _comp[_iter.last_index] = _call[2]
                    try:
                        _tmp = _comp.__getitem__(_call[1])
                    except KeyError:
                        if self._slice.key_errors_generate_exceptions and (not self._iter_over_index):
                            raise
                        break
                    if _tmp.__class__ is IndexedComponent_slice:
                        assert _tmp._len == 1
                        _iter = _IndexedComponent_slice_iter(_tmp, self.advance_iter, sort=self._sort)
                        for _ in _iter:
                            self.advance_iter.check_complete()
                            _comp[_iter.get_last_index()] = _call[2]
                        break
                    else:
                        self.advance_iter.check_complete()
                        _comp[_call[1]] = _call[2]
                elif _call[0] == IndexedComponent_slice.del_item:
                    assert idx == self._slice._len - 1
                    try:
                        _tmp = _comp.__getitem__(_call[1])
                    except KeyError:
                        if self._slice.key_errors_generate_exceptions:
                            raise
                        break
                    if _tmp.__class__ is IndexedComponent_slice:
                        assert _tmp._len == 1
                        _iter = _IndexedComponent_slice_iter(_tmp, self.advance_iter, sort=self._sort)
                        _idx_to_del = []
                        for _ in _iter:
                            _idx_to_del.append(_iter.get_last_index())
                        self.advance_iter.check_complete()
                        for _idx in _idx_to_del:
                            del _comp[_idx]
                        break
                    else:
                        del _comp[_call[1]]
                elif _call[0] == IndexedComponent_slice.del_attribute:
                    assert idx == self._slice._len - 1
                    try:
                        _comp = delattr(_comp, _call[1])
                    except AttributeError:
                        if self._slice.attribute_errors_generate_exceptions:
                            raise
                        break
                else:
                    raise DeveloperError('Unexpected entry in IndexedComponent_slice _call_stack: %s' % (_call[0],))
                idx += 1
            if idx == self._slice._len:
                self.advance_iter.check_complete()
                return _comp

    def get_last_index(self):
        ans = sum((x.last_index for x in self._iter_stack if x is not None), ())
        if len(ans) == 1:
            return ans[0]
        else:
            return ans

    def get_last_index_wildcards(self):
        """Get a tuple of the values in the wildcard positions for the most
        recent indices corresponding to the last component returned by
        each _slice_generator in the iter stack.

        """
        ans = sum((tuple((x.last_index[i] for i in range(len(x.last_index)) if i not in x.fixed)) for x in self._iter_stack if x is not None), ())
        if not ans:
            return UnindexedComponent_index
        if len(ans) == 1:
            return ans[0]
        else:
            return ans