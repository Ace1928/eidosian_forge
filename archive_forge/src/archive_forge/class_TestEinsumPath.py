import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
class TestEinsumPath:

    def build_operands(self, string, size_dict=global_size_dict):
        operands = [string]
        terms = string.split('->')[0].split(',')
        for term in terms:
            dims = [size_dict[x] for x in term]
            operands.append(np.random.rand(*dims))
        return operands

    def assert_path_equal(self, comp, benchmark):
        ret = len(comp) == len(benchmark)
        assert_(ret)
        for pos in range(len(comp) - 1):
            ret &= isinstance(comp[pos + 1], tuple)
            ret &= comp[pos + 1] == benchmark[pos + 1]
        assert_(ret)

    def test_memory_contraints(self):
        outer_test = self.build_operands('a,b,c->abc')
        path, path_str = np.einsum_path(*outer_test, optimize=('greedy', 0))
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2)])
        path, path_str = np.einsum_path(*outer_test, optimize=('optimal', 0))
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2)])
        long_test = self.build_operands('acdf,jbje,gihb,hfac')
        path, path_str = np.einsum_path(*long_test, optimize=('greedy', 0))
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2, 3)])
        path, path_str = np.einsum_path(*long_test, optimize=('optimal', 0))
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2, 3)])

    def test_long_paths(self):
        long_test1 = self.build_operands('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        path, path_str = np.einsum_path(*long_test1, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (3, 6), (3, 4), (2, 4), (2, 3), (0, 2), (0, 1)])
        path, path_str = np.einsum_path(*long_test1, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (3, 6), (3, 4), (2, 4), (2, 3), (0, 2), (0, 1)])
        long_test2 = self.build_operands('chd,bde,agbc,hiad,bdi,cgh,agdb')
        path, path_str = np.einsum_path(*long_test2, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (3, 4), (0, 3), (3, 4), (1, 3), (1, 2), (0, 1)])
        path, path_str = np.einsum_path(*long_test2, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (0, 5), (1, 4), (3, 4), (1, 3), (1, 2), (0, 1)])

    def test_edge_paths(self):
        edge_test1 = self.build_operands('eb,cb,fb->cef')
        path, path_str = np.einsum_path(*edge_test1, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (0, 2), (0, 1)])
        path, path_str = np.einsum_path(*edge_test1, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (0, 2), (0, 1)])
        edge_test2 = self.build_operands('dd,fb,be,cdb->cef')
        path, path_str = np.einsum_path(*edge_test2, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (0, 3), (0, 1), (0, 1)])
        path, path_str = np.einsum_path(*edge_test2, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (0, 3), (0, 1), (0, 1)])
        edge_test3 = self.build_operands('bca,cdb,dbf,afc->')
        path, path_str = np.einsum_path(*edge_test3, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 2), (0, 1)])
        path, path_str = np.einsum_path(*edge_test3, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 2), (0, 1)])
        edge_test4 = self.build_operands('dcc,fce,ea,dbf->ab')
        path, path_str = np.einsum_path(*edge_test4, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 1), (0, 1)])
        path, path_str = np.einsum_path(*edge_test4, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 2), (0, 1)])
        edge_test4 = self.build_operands('a,ac,ab,ad,cd,bd,bc->', size_dict={'a': 20, 'b': 20, 'c': 20, 'd': 20})
        path, path_str = np.einsum_path(*edge_test4, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (0, 1), (0, 1, 2, 3, 4, 5)])
        path, path_str = np.einsum_path(*edge_test4, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (0, 1), (0, 1, 2, 3, 4, 5)])

    def test_path_type_input(self):
        path_test = self.build_operands('dcc,fce,ea,dbf->ab')
        path, path_str = np.einsum_path(*path_test, optimize=False)
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2, 3)])
        path, path_str = np.einsum_path(*path_test, optimize=True)
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 1), (0, 1)])
        exp_path = ['einsum_path', (0, 2), (0, 2), (0, 1)]
        path, path_str = np.einsum_path(*path_test, optimize=exp_path)
        self.assert_path_equal(path, exp_path)
        noopt = np.einsum(*path_test, optimize=False)
        opt = np.einsum(*path_test, optimize=exp_path)
        assert_almost_equal(noopt, opt)

    def test_path_type_input_internal_trace(self):
        path_test = self.build_operands('cab,cdd->ab')
        exp_path = ['einsum_path', (1,), (0, 1)]
        path, path_str = np.einsum_path(*path_test, optimize=exp_path)
        self.assert_path_equal(path, exp_path)
        noopt = np.einsum(*path_test, optimize=False)
        opt = np.einsum(*path_test, optimize=exp_path)
        assert_almost_equal(noopt, opt)

    def test_path_type_input_invalid(self):
        path_test = self.build_operands('ab,bc,cd,de->ae')
        exp_path = ['einsum_path', (2, 3), (0, 1)]
        assert_raises(RuntimeError, np.einsum, *path_test, optimize=exp_path)
        assert_raises(RuntimeError, np.einsum_path, *path_test, optimize=exp_path)
        path_test = self.build_operands('a,a,a->a')
        exp_path = ['einsum_path', (1,), (0, 1)]
        assert_raises(RuntimeError, np.einsum, *path_test, optimize=exp_path)
        assert_raises(RuntimeError, np.einsum_path, *path_test, optimize=exp_path)

    def test_spaces(self):
        arr = np.array([[1]])
        for sp in itertools.product(['', ' '], repeat=4):
            np.einsum('{}...a{}->{}...a{}'.format(*sp), arr)