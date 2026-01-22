import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
class TestMultiIndexingAutomated:
    """
    These tests use code to mimic the C-Code indexing for selection.

    NOTE:

        * This still lacks tests for complex item setting.
        * If you change behavior of indexing, you might want to modify
          these tests to try more combinations.
        * Behavior was written to match numpy version 1.8. (though a
          first version matched 1.7.)
        * Only tuple indices are supported by the mimicking code.
          (and tested as of writing this)
        * Error types should match most of the time as long as there
          is only one error. For multiple errors, what gets raised
          will usually not be the same one. They are *not* tested.

    Update 2016-11-30: It is probably not worth maintaining this test
    indefinitely and it can be dropped if maintenance becomes a burden.

    """

    def setup_method(self):
        self.a = np.arange(np.prod([3, 1, 5, 6])).reshape(3, 1, 5, 6)
        self.b = np.empty((3, 0, 5, 6))
        self.complex_indices = ['skip', Ellipsis, 0, np.array([True, False, False]), np.array([[True, False], [False, True]]), np.array([[[False, False], [False, False]]]), slice(-5, 5, 2), slice(1, 1, 100), slice(4, -1, -2), slice(None, None, -3), np.empty((0, 1, 1), dtype=np.intp), np.array([0, 1, -2]), np.array([[2], [0], [1]]), np.array([[0, -1], [0, 1]], dtype=np.dtype('intp').newbyteorder()), np.array([2, -1], dtype=np.int8), np.zeros([1] * 31, dtype=int), np.array([0.0, 1.0])]
        self.simple_indices = [Ellipsis, None, -1, [1], np.array([True]), 'skip']
        self.fill_indices = [slice(None, None), 0]

    def _get_multi_index(self, arr, indices):
        """Mimic multi dimensional indexing.

        Parameters
        ----------
        arr : ndarray
            Array to be indexed.
        indices : tuple of index objects

        Returns
        -------
        out : ndarray
            An array equivalent to the indexing operation (but always a copy).
            `arr[indices]` should be identical.
        no_copy : bool
            Whether the indexing operation requires a copy. If this is `True`,
            `np.may_share_memory(arr, arr[indices])` should be `True` (with
            some exceptions for scalars and possibly 0-d arrays).

        Notes
        -----
        While the function may mostly match the errors of normal indexing this
        is generally not the case.
        """
        in_indices = list(indices)
        indices = []
        no_copy = True
        num_fancy = 0
        fancy_dim = 0
        error_unless_broadcast_to_empty = False
        ndim = 0
        ellipsis_pos = None
        for i, indx in enumerate(in_indices):
            if indx is None:
                continue
            if isinstance(indx, np.ndarray) and indx.dtype == bool:
                no_copy = False
                if indx.ndim == 0:
                    raise IndexError
                ndim += indx.ndim
                fancy_dim += indx.ndim
                continue
            if indx is Ellipsis:
                if ellipsis_pos is None:
                    ellipsis_pos = i
                    continue
                raise IndexError
            if isinstance(indx, slice):
                ndim += 1
                continue
            if not isinstance(indx, np.ndarray):
                try:
                    indx = np.array(indx, dtype=np.intp)
                except ValueError:
                    raise IndexError
                in_indices[i] = indx
            elif indx.dtype.kind != 'b' and indx.dtype.kind != 'i':
                raise IndexError('arrays used as indices must be of integer (or boolean) type')
            if indx.ndim != 0:
                no_copy = False
            ndim += 1
            fancy_dim += 1
        if arr.ndim - ndim < 0:
            raise IndexError
        if ndim == 0 and None not in in_indices:
            return (arr.copy(), no_copy)
        if ellipsis_pos is not None:
            in_indices[ellipsis_pos:ellipsis_pos + 1] = [slice(None, None)] * (arr.ndim - ndim)
        for ax, indx in enumerate(in_indices):
            if isinstance(indx, slice):
                indx = np.arange(*indx.indices(arr.shape[ax]))
                indices.append(['s', indx])
                continue
            elif indx is None:
                indices.append(['n', np.array([0], dtype=np.intp)])
                arr = arr.reshape(arr.shape[:ax] + (1,) + arr.shape[ax:])
                continue
            if isinstance(indx, np.ndarray) and indx.dtype == bool:
                if indx.shape != arr.shape[ax:ax + indx.ndim]:
                    raise IndexError
                try:
                    flat_indx = np.ravel_multi_index(np.nonzero(indx), arr.shape[ax:ax + indx.ndim], mode='raise')
                except Exception:
                    error_unless_broadcast_to_empty = True
                    flat_indx = np.array([0] * indx.sum(), dtype=np.intp)
                if indx.ndim != 0:
                    arr = arr.reshape(arr.shape[:ax] + (np.prod(arr.shape[ax:ax + indx.ndim]),) + arr.shape[ax + indx.ndim:])
                    indx = flat_indx
                else:
                    raise IndexError
            elif indx.ndim == 0:
                if indx >= arr.shape[ax] or indx < -arr.shape[ax]:
                    raise IndexError
            if indx.ndim == 0:
                if indx >= arr.shape[ax] or indx < -arr.shape[ax]:
                    raise IndexError
            if len(indices) > 0 and indices[-1][0] == 'f' and (ax != ellipsis_pos):
                indices[-1].append(indx)
            else:
                num_fancy += 1
                indices.append(['f', indx])
        if num_fancy > 1 and (not no_copy):
            new_indices = indices[:]
            axes = list(range(arr.ndim))
            fancy_axes = []
            new_indices.insert(0, ['f'])
            ni = 0
            ai = 0
            for indx in indices:
                ni += 1
                if indx[0] == 'f':
                    new_indices[0].extend(indx[1:])
                    del new_indices[ni]
                    ni -= 1
                    for ax in range(ai, ai + len(indx[1:])):
                        fancy_axes.append(ax)
                        axes.remove(ax)
                ai += len(indx) - 1
            indices = new_indices
            arr = arr.transpose(*fancy_axes + axes)
        ax = 0
        for indx in indices:
            if indx[0] == 'f':
                if len(indx) == 1:
                    continue
                orig_shape = arr.shape
                orig_slice = orig_shape[ax:ax + len(indx[1:])]
                arr = arr.reshape(arr.shape[:ax] + (np.prod(orig_slice).astype(int),) + arr.shape[ax + len(indx[1:]):])
                res = np.broadcast(*indx[1:])
                if res.size != 0:
                    if error_unless_broadcast_to_empty:
                        raise IndexError
                    for _indx, _size in zip(indx[1:], orig_slice):
                        if _indx.size == 0:
                            continue
                        if np.any(_indx >= _size) or np.any(_indx < -_size):
                            raise IndexError
                if len(indx[1:]) == len(orig_slice):
                    if np.prod(orig_slice) == 0:
                        try:
                            mi = np.ravel_multi_index(indx[1:], orig_slice, mode='raise')
                        except Exception:
                            raise IndexError('invalid index into 0-sized')
                    else:
                        mi = np.ravel_multi_index(indx[1:], orig_slice, mode='wrap')
                else:
                    raise ValueError
                arr = arr.take(mi.ravel(), axis=ax)
                try:
                    arr = arr.reshape(arr.shape[:ax] + mi.shape + arr.shape[ax + 1:])
                except ValueError:
                    raise IndexError
                ax += mi.ndim
                continue
            arr = arr.take(indx[1], axis=ax)
            ax += 1
        return (arr, no_copy)

    def _check_multi_index(self, arr, index):
        """Check a multi index item getting and simple setting.

        Parameters
        ----------
        arr : ndarray
            Array to be indexed, must be a reshaped arange.
        index : tuple of indexing objects
            Index being tested.
        """
        try:
            mimic_get, no_copy = self._get_multi_index(arr, index)
        except Exception as e:
            if HAS_REFCOUNT:
                prev_refcount = sys.getrefcount(arr)
            assert_raises(type(e), arr.__getitem__, index)
            assert_raises(type(e), arr.__setitem__, index, 0)
            if HAS_REFCOUNT:
                assert_equal(prev_refcount, sys.getrefcount(arr))
            return
        self._compare_index_result(arr, index, mimic_get, no_copy)

    def _check_single_index(self, arr, index):
        """Check a single index item getting and simple setting.

        Parameters
        ----------
        arr : ndarray
            Array to be indexed, must be an arange.
        index : indexing object
            Index being tested. Must be a single index and not a tuple
            of indexing objects (see also `_check_multi_index`).
        """
        try:
            mimic_get, no_copy = self._get_multi_index(arr, (index,))
        except Exception as e:
            if HAS_REFCOUNT:
                prev_refcount = sys.getrefcount(arr)
            assert_raises(type(e), arr.__getitem__, index)
            assert_raises(type(e), arr.__setitem__, index, 0)
            if HAS_REFCOUNT:
                assert_equal(prev_refcount, sys.getrefcount(arr))
            return
        self._compare_index_result(arr, index, mimic_get, no_copy)

    def _compare_index_result(self, arr, index, mimic_get, no_copy):
        """Compare mimicked result to indexing result.
        """
        arr = arr.copy()
        indexed_arr = arr[index]
        assert_array_equal(indexed_arr, mimic_get)
        if indexed_arr.size != 0 and indexed_arr.ndim != 0:
            assert_(np.may_share_memory(indexed_arr, arr) == no_copy)
            if HAS_REFCOUNT:
                if no_copy:
                    assert_equal(sys.getrefcount(arr), 3)
                else:
                    assert_equal(sys.getrefcount(arr), 2)
        b = arr.copy()
        b[index] = mimic_get + 1000
        if b.size == 0:
            return
        if no_copy and indexed_arr.ndim != 0:
            indexed_arr += 1000
            assert_array_equal(arr, b)
            return
        arr.flat[indexed_arr.ravel()] += 1000
        assert_array_equal(arr, b)

    def test_boolean(self):
        a = np.array(5)
        assert_equal(a[np.array(True)], 5)
        a[np.array(True)] = 1
        assert_equal(a, 1)
        self._check_multi_index(self.a, (np.zeros_like(self.a, dtype=bool),))
        self._check_multi_index(self.a, (np.zeros_like(self.a, dtype=bool)[..., 0],))
        self._check_multi_index(self.a, (np.zeros_like(self.a, dtype=bool)[None, ...],))

    def test_multidim(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('error', '', DeprecationWarning)
            warnings.filterwarnings('error', '', np.VisibleDeprecationWarning)

            def isskip(idx):
                return isinstance(idx, str) and idx == 'skip'
            for simple_pos in [0, 2, 3]:
                tocheck = [self.fill_indices, self.complex_indices, self.fill_indices, self.fill_indices]
                tocheck[simple_pos] = self.simple_indices
                for index in product(*tocheck):
                    index = tuple((i for i in index if not isskip(i)))
                    self._check_multi_index(self.a, index)
                    self._check_multi_index(self.b, index)
        self._check_multi_index(self.a, (0, 0, 0, 0))
        self._check_multi_index(self.b, (0, 0, 0, 0))
        assert_raises(IndexError, self.a.__getitem__, (0, 0, 0, 0, 0))
        assert_raises(IndexError, self.a.__setitem__, (0, 0, 0, 0, 0), 0)
        assert_raises(IndexError, self.a.__getitem__, (0, 0, [1], 0, 0))
        assert_raises(IndexError, self.a.__setitem__, (0, 0, [1], 0, 0), 0)

    def test_1d(self):
        a = np.arange(10)
        for index in self.complex_indices:
            self._check_single_index(a, index)