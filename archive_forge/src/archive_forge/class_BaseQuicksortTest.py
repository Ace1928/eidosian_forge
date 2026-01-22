import copy
import itertools
import math
import random
import sys
import unittest
import numpy as np
from numba import jit, njit
from numba.core import utils, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.misc.quicksort import make_py_quicksort, make_jit_quicksort
from numba.misc.mergesort import make_jit_mergesort
from numba.misc.timsort import make_py_timsort, make_jit_timsort, MergeRun
class BaseQuicksortTest(BaseSortingTest):

    def test_insertion_sort(self):
        n = 20

        def check(l, n):
            res = self.array_factory([9999] + l + [-9999])
            f(res, res, 1, n)
            self.assertEqual(res[0], 9999)
            self.assertEqual(res[-1], -9999)
            self.assertSorted(l, res[1:-1])
        f = self.quicksort.insertion_sort
        l = self.sorted_list(n)
        check(l, n)
        l = self.revsorted_list(n)
        check(l, n)
        l = self.initially_sorted_list(n, n // 2)
        check(l, n)
        l = self.revsorted_list(n)
        check(l, n)
        l = self.random_list(n)
        check(l, n)
        l = self.duprandom_list(n)
        check(l, n)

    def test_partition(self):
        n = 20

        def check(l, n):
            res = self.array_factory([9999] + l + [-9999])
            index = f(res, res, 1, n)
            self.assertEqual(res[0], 9999)
            self.assertEqual(res[-1], -9999)
            pivot = res[index]
            for i in range(1, index):
                self.assertLessEqual(res[i], pivot)
            for i in range(index + 1, n):
                self.assertGreaterEqual(res[i], pivot)
        f = self.quicksort.partition
        l = self.sorted_list(n)
        check(l, n)
        l = self.revsorted_list(n)
        check(l, n)
        l = self.initially_sorted_list(n, n // 2)
        check(l, n)
        l = self.revsorted_list(n)
        check(l, n)
        l = self.random_list(n)
        check(l, n)
        l = self.duprandom_list(n)
        check(l, n)

    def test_partition3(self):
        n = 20

        def check(l, n):
            res = self.array_factory([9999] + l + [-9999])
            lt, gt = f(res, 1, n)
            self.assertEqual(res[0], 9999)
            self.assertEqual(res[-1], -9999)
            pivot = res[lt]
            for i in range(1, lt):
                self.assertLessEqual(res[i], pivot)
            for i in range(lt, gt + 1):
                self.assertEqual(res[i], pivot)
            for i in range(gt + 1, n):
                self.assertGreater(res[i], pivot)
        f = self.quicksort.partition3
        l = self.sorted_list(n)
        check(l, n)
        l = self.revsorted_list(n)
        check(l, n)
        l = self.initially_sorted_list(n, n // 2)
        check(l, n)
        l = self.revsorted_list(n)
        check(l, n)
        l = self.random_list(n)
        check(l, n)
        l = self.duprandom_list(n)
        check(l, n)

    def test_run_quicksort(self):
        f = self.quicksort.run_quicksort
        for size_factor in (1, 5):
            sizes = (15, 20)
            all_lists = [self.make_sample_lists(n * size_factor) for n in sizes]
            for chunks in itertools.product(*all_lists):
                orig_keys = sum(chunks, [])
                keys = self.array_factory(orig_keys)
                f(keys)
                self.assertSorted(orig_keys, keys)

    def test_run_quicksort_lt(self):

        def lt(a, b):
            return a > b
        f = self.make_quicksort(lt=lt).run_quicksort
        for size_factor in (1, 5):
            sizes = (15, 20)
            all_lists = [self.make_sample_lists(n * size_factor) for n in sizes]
            for chunks in itertools.product(*all_lists):
                orig_keys = sum(chunks, [])
                keys = self.array_factory(orig_keys)
                f(keys)
                self.assertSorted(orig_keys, keys[::-1])

        def lt_floats(a, b):
            return math.isnan(b) or a < b
        f = self.make_quicksort(lt=lt_floats).run_quicksort
        np.random.seed(42)
        for size in (5, 20, 50, 500):
            orig = np.random.random(size=size) * 100
            orig[np.random.random(size=size) < 0.1] = float('nan')
            orig_keys = list(orig)
            keys = self.array_factory(orig_keys)
            f(keys)
            non_nans = orig[~np.isnan(orig)]
            self.assertSorted(non_nans, keys[:len(non_nans)])