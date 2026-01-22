from contextlib import nullcontext
import itertools
import locale
import logging
import re
from packaging.version import parse as parse_version
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
class TestAutoMinorLocator:

    def test_basic(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1.39)
        ax.minorticks_on()
        test_value = np.array([0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.45, 0.5, 0.55, 0.65, 0.7, 0.75, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.25, 1.3, 1.35])
        assert_almost_equal(ax.xaxis.get_ticklocs(minor=True), test_value)
    params = [(0, 0), (1, 0)]

    def test_first_and_last_minorticks(self):
        """
        Test that first and last minor tick appear as expected.
        """
        fig, ax = plt.subplots()
        ax.set_xlim(-1.9, 1.9)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        test_value = np.array([-1.9, -1.8, -1.7, -1.6, -1.4, -1.3, -1.2, -1.1, -0.9, -0.8, -0.7, -0.6, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 1.8, 1.9])
        assert_almost_equal(ax.xaxis.get_ticklocs(minor=True), test_value)
        ax.set_xlim(-5, 5)
        test_value = np.array([-5.0, -4.5, -3.5, -3.0, -2.5, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.5, 3.0, 3.5, 4.5, 5.0])
        assert_almost_equal(ax.xaxis.get_ticklocs(minor=True), test_value)

    @pytest.mark.parametrize('nb_majorticks, expected_nb_minorticks', params)
    def test_low_number_of_majorticks(self, nb_majorticks, expected_nb_minorticks):
        fig, ax = plt.subplots()
        xlims = (0, 5)
        ax.set_xlim(*xlims)
        ax.set_xticks(np.linspace(xlims[0], xlims[1], nb_majorticks))
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        assert len(ax.xaxis.get_minorticklocs()) == expected_nb_minorticks
    majorstep_minordivisions = [(1, 5), (2, 4), (2.5, 5), (5, 5), (10, 5)]

    def test_using_all_default_major_steps(self):
        with mpl.rc_context({'_internal.classic_mode': False}):
            majorsteps = [x[0] for x in self.majorstep_minordivisions]
            np.testing.assert_allclose(majorsteps, mticker.AutoLocator()._steps)

    @pytest.mark.parametrize('major_step, expected_nb_minordivisions', majorstep_minordivisions)
    def test_number_of_minor_ticks(self, major_step, expected_nb_minordivisions):
        fig, ax = plt.subplots()
        xlims = (0, major_step)
        ax.set_xlim(*xlims)
        ax.set_xticks(xlims)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        nb_minor_divisions = len(ax.xaxis.get_minorticklocs()) + 1
        assert nb_minor_divisions == expected_nb_minordivisions
    limits = [(0, 1.39), (0, 0.139), (0, 1.1e-20), (0, 1.12e-13), (-2e-07, -3.3e-08), (1.2e-06, 1.42e-06), (-1.34e-06, -1.44e-06), (-8.76e-07, -1.51e-06)]
    reference = [[0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.45, 0.5, 0.55, 0.65, 0.7, 0.75, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.25, 1.3, 1.35], [0.005, 0.01, 0.015, 0.025, 0.03, 0.035, 0.045, 0.05, 0.055, 0.065, 0.07, 0.075, 0.085, 0.09, 0.095, 0.105, 0.11, 0.115, 0.125, 0.13, 0.135], [5e-22, 1e-21, 1.5e-21, 2.5e-21, 3e-21, 3.5e-21, 4.5e-21, 5e-21, 5.5e-21, 6.5e-21, 7e-21, 7.5e-21, 8.5e-21, 9e-21, 9.5e-21, 1.05e-20, 1.1e-20], [5e-15, 1e-14, 1.5e-14, 2.5e-14, 3e-14, 3.5e-14, 4.5e-14, 5e-14, 5.5e-14, 6.5e-14, 7e-14, 7.5e-14, 8.5e-14, 9e-14, 9.5e-14, 1.05e-13, 1.1e-13], [-1.95e-07, -1.9e-07, -1.85e-07, -1.75e-07, -1.7e-07, -1.65e-07, -1.55e-07, -1.5e-07, -1.45e-07, -1.35e-07, -1.3e-07, -1.25e-07, -1.15e-07, -1.1e-07, -1.05e-07, -9.5e-08, -9e-08, -8.5e-08, -7.5e-08, -7e-08, -6.5e-08, -5.5e-08, -5e-08, -4.5e-08, -3.5e-08], [1.21e-06, 1.22e-06, 1.23e-06, 1.24e-06, 1.26e-06, 1.27e-06, 1.28e-06, 1.29e-06, 1.31e-06, 1.32e-06, 1.33e-06, 1.34e-06, 1.36e-06, 1.37e-06, 1.38e-06, 1.39e-06, 1.41e-06, 1.42e-06], [-1.435e-06, -1.43e-06, -1.425e-06, -1.415e-06, -1.41e-06, -1.405e-06, -1.395e-06, -1.39e-06, -1.385e-06, -1.375e-06, -1.37e-06, -1.365e-06, -1.355e-06, -1.35e-06, -1.345e-06], [-1.48e-06, -1.46e-06, -1.44e-06, -1.42e-06, -1.38e-06, -1.36e-06, -1.34e-06, -1.32e-06, -1.28e-06, -1.26e-06, -1.24e-06, -1.22e-06, -1.18e-06, -1.16e-06, -1.14e-06, -1.12e-06, -1.08e-06, -1.06e-06, -1.04e-06, -1.02e-06, -9.8e-07, -9.6e-07, -9.4e-07, -9.2e-07, -8.8e-07]]
    additional_data = list(zip(limits, reference))

    @pytest.mark.parametrize('lim, ref', additional_data)
    def test_additional(self, lim, ref):
        fig, ax = plt.subplots()
        ax.minorticks_on()
        ax.grid(True, 'minor', 'y', linewidth=1)
        ax.grid(True, 'major', color='k', linewidth=1)
        ax.set_ylim(lim)
        assert_almost_equal(ax.yaxis.get_ticklocs(minor=True), ref)

    @pytest.mark.parametrize('use_rcparam', [False, True])
    @pytest.mark.parametrize('lim, ref', [((0, 1.39), [0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.45, 0.5, 0.55, 0.65, 0.7, 0.75, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.25, 1.3, 1.35]), ((0, 0.139), [0.005, 0.01, 0.015, 0.025, 0.03, 0.035, 0.045, 0.05, 0.055, 0.065, 0.07, 0.075, 0.085, 0.09, 0.095, 0.105, 0.11, 0.115, 0.125, 0.13, 0.135])])
    def test_number_of_minor_ticks_auto(self, lim, ref, use_rcparam):
        if use_rcparam:
            context = {'xtick.minor.ndivs': 'auto', 'ytick.minor.ndivs': 'auto'}
            kwargs = {}
        else:
            context = {}
            kwargs = {'n': 'auto'}
        with mpl.rc_context(context):
            fig, ax = plt.subplots()
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
            ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(**kwargs))
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(**kwargs))
            assert_almost_equal(ax.xaxis.get_ticklocs(minor=True), ref)
            assert_almost_equal(ax.yaxis.get_ticklocs(minor=True), ref)

    @pytest.mark.parametrize('use_rcparam', [False, True])
    @pytest.mark.parametrize('n, lim, ref', [(2, (0, 4), [0.5, 1.5, 2.5, 3.5]), (4, (0, 2), [0.25, 0.5, 0.75, 1.25, 1.5, 1.75]), (10, (0, 1), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])])
    def test_number_of_minor_ticks_int(self, n, lim, ref, use_rcparam):
        if use_rcparam:
            context = {'xtick.minor.ndivs': n, 'ytick.minor.ndivs': n}
            kwargs = {}
        else:
            context = {}
            kwargs = {'n': n}
        with mpl.rc_context(context):
            fig, ax = plt.subplots()
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
            ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(**kwargs))
            ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(**kwargs))
            assert_almost_equal(ax.xaxis.get_ticklocs(minor=True), ref)
            assert_almost_equal(ax.yaxis.get_ticklocs(minor=True), ref)