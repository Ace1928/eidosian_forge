import warnings
import re
import sys
import pickle
from pathlib import Path
import os
import json
import platform
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import numpy
import numpy as np
from numpy import typecodes, array
from numpy.lib.recfunctions import rec_append_fields
from scipy import special
from scipy._lib._util import check_random_state
from scipy.integrate import (IntegrationWarning, quad, trapezoid,
import scipy.stats as stats
from scipy.stats._distn_infrastructure import argsreduce
import scipy.stats.distributions
from scipy.special import xlogy, polygamma, entr
from scipy.stats._distr_params import distcont, invdistcont
from .test_discrete_basic import distdiscrete, invdistdiscrete
from scipy.stats._continuous_distns import FitDataError, _argus_phi
from scipy.optimize import root, fmin, differential_evolution
from itertools import product
class TestSubclassingExplicitShapes:

    def test_correct_shapes(self):
        dummy_distr = _distr_gen(name='dummy', shapes='a')
        assert_equal(dummy_distr.pdf(1, a=1), 42)

    def test_wrong_shapes_1(self):
        dummy_distr = _distr_gen(name='dummy', shapes='A')
        assert_raises(TypeError, dummy_distr.pdf, 1, **dict(a=1))

    def test_wrong_shapes_2(self):
        dummy_distr = _distr_gen(name='dummy', shapes='a, b, c')
        dct = dict(a=1, b=2, c=3)
        assert_raises(TypeError, dummy_distr.pdf, 1, **dct)

    def test_shapes_string(self):
        dct = dict(name='dummy', shapes=42)
        assert_raises(TypeError, _distr_gen, **dct)

    def test_shapes_identifiers_1(self):
        dct = dict(name='dummy', shapes='(!)')
        assert_raises(SyntaxError, _distr_gen, **dct)

    def test_shapes_identifiers_2(self):
        dct = dict(name='dummy', shapes='4chan')
        assert_raises(SyntaxError, _distr_gen, **dct)

    def test_shapes_identifiers_3(self):
        dct = dict(name='dummy', shapes='m(fti)')
        assert_raises(SyntaxError, _distr_gen, **dct)

    def test_shapes_identifiers_nodefaults(self):
        dct = dict(name='dummy', shapes='a=2')
        assert_raises(SyntaxError, _distr_gen, **dct)

    def test_shapes_args(self):
        dct = dict(name='dummy', shapes='*args')
        assert_raises(SyntaxError, _distr_gen, **dct)

    def test_shapes_kwargs(self):
        dct = dict(name='dummy', shapes='**kwargs')
        assert_raises(SyntaxError, _distr_gen, **dct)

    def test_shapes_keywords(self):
        dct = dict(name='dummy', shapes='a, b, c, lambda')
        assert_raises(SyntaxError, _distr_gen, **dct)

    def test_shapes_signature(self):

        class _dist_gen(stats.rv_continuous):

            def _pdf(self, x, a):
                return stats.norm._pdf(x) * a
        dist = _dist_gen(shapes='a')
        assert_equal(dist.pdf(0.5, a=2), stats.norm.pdf(0.5) * 2)

    def test_shapes_signature_inconsistent(self):

        class _dist_gen(stats.rv_continuous):

            def _pdf(self, x, a):
                return stats.norm._pdf(x) * a
        dist = _dist_gen(shapes='a, b')
        assert_raises(TypeError, dist.pdf, 0.5, **dict(a=1, b=2))

    def test_star_args(self):

        class _dist_gen(stats.rv_continuous):

            def _pdf(self, x, *args):
                extra_kwarg = args[0]
                return stats.norm._pdf(x) * extra_kwarg
        dist = _dist_gen(shapes='extra_kwarg')
        assert_equal(dist.pdf(0.5, extra_kwarg=33), stats.norm.pdf(0.5) * 33)
        assert_equal(dist.pdf(0.5, 33), stats.norm.pdf(0.5) * 33)
        assert_raises(TypeError, dist.pdf, 0.5, **dict(xxx=33))

    def test_star_args_2(self):

        class _dist_gen(stats.rv_continuous):

            def _pdf(self, x, offset, *args):
                extra_kwarg = args[0]
                return stats.norm._pdf(x) * extra_kwarg + offset
        dist = _dist_gen(shapes='offset, extra_kwarg')
        assert_equal(dist.pdf(0.5, offset=111, extra_kwarg=33), stats.norm.pdf(0.5) * 33 + 111)
        assert_equal(dist.pdf(0.5, 111, 33), stats.norm.pdf(0.5) * 33 + 111)

    def test_extra_kwarg(self):

        class _distr_gen(stats.rv_continuous):

            def _pdf(self, x, *args, **kwargs):
                extra_kwarg = kwargs.pop('extra_kwarg', 1)
                return stats.norm._pdf(x) * extra_kwarg
        dist = _distr_gen(shapes='extra_kwarg')
        assert_equal(dist.pdf(1, extra_kwarg=3), stats.norm.pdf(1))

    def test_shapes_empty_string(self):

        class _dist_gen(stats.rv_continuous):

            def _pdf(self, x):
                return stats.norm.pdf(x)
        dist = _dist_gen(shapes='')
        assert_equal(dist.pdf(0.5), stats.norm.pdf(0.5))