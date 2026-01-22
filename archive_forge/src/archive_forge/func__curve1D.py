from pyomo.common.dependencies import numpy as np, pandas as pd, matplotlib as plt
from pyomo.core.expr.numvalue import value
from itertools import product
import logging
from pyomo.opt import SolverStatus, TerminationCondition
def _curve1D(self, title_text, xlabel_text, font_axes=16, font_tick=14, log_scale=True):
    """
        Draw 1D sensitivity curves for all design criteria

        Parameters
        ----------
        title_text: name of the figure, a string
        xlabel_text: x label title, a string.
            In a 1D sensitivity curve, it is the design variable by which the curve is drawn.
        font_axes: axes label font size
        font_tick: tick label font size
        log_scale: if True, the result matrix will be scaled by log10

        Returns
        --------
        4 Figures of 1D sensitivity curves for each criteria
        """
    x_range = self.figure_result_data[self.sensitivity_dimension[0]].values.tolist()
    if log_scale:
        y_range_A = np.log10(self.figure_result_data['A'].values.tolist())
        y_range_D = np.log10(self.figure_result_data['D'].values.tolist())
        y_range_E = np.log10(self.figure_result_data['E'].values.tolist())
        y_range_ME = np.log10(self.figure_result_data['ME'].values.tolist())
    else:
        y_range_A = self.figure_result_data['A'].values.tolist()
        y_range_D = self.figure_result_data['D'].values.tolist()
        y_range_E = self.figure_result_data['E'].values.tolist()
        y_range_ME = self.figure_result_data['ME'].values.tolist()
    fig = plt.pyplot.figure()
    plt.pyplot.rc('axes', titlesize=font_axes)
    plt.pyplot.rc('axes', labelsize=font_axes)
    plt.pyplot.rc('xtick', labelsize=font_tick)
    plt.pyplot.rc('ytick', labelsize=font_tick)
    ax = fig.add_subplot(111)
    params = {'mathtext.default': 'regular'}
    ax.plot(x_range, y_range_A)
    ax.scatter(x_range, y_range_A)
    ax.set_ylabel('$log_{10}$ Trace')
    ax.set_xlabel(xlabel_text)
    plt.pyplot.title(title_text + ' - A optimality')
    plt.pyplot.show()
    fig = plt.pyplot.figure()
    plt.pyplot.rc('axes', titlesize=font_axes)
    plt.pyplot.rc('axes', labelsize=font_axes)
    plt.pyplot.rc('xtick', labelsize=font_tick)
    plt.pyplot.rc('ytick', labelsize=font_tick)
    ax = fig.add_subplot(111)
    params = {'mathtext.default': 'regular'}
    ax.plot(x_range, y_range_D)
    ax.scatter(x_range, y_range_D)
    ax.set_ylabel('$log_{10}$ Determinant')
    ax.set_xlabel(xlabel_text)
    plt.pyplot.title(title_text + ' - D optimality')
    plt.pyplot.show()
    fig = plt.pyplot.figure()
    plt.pyplot.rc('axes', titlesize=font_axes)
    plt.pyplot.rc('axes', labelsize=font_axes)
    plt.pyplot.rc('xtick', labelsize=font_tick)
    plt.pyplot.rc('ytick', labelsize=font_tick)
    ax = fig.add_subplot(111)
    params = {'mathtext.default': 'regular'}
    ax.plot(x_range, y_range_E)
    ax.scatter(x_range, y_range_E)
    ax.set_ylabel('$log_{10}$ Minimal eigenvalue')
    ax.set_xlabel(xlabel_text)
    plt.pyplot.title(title_text + ' - E optimality')
    plt.pyplot.show()
    fig = plt.pyplot.figure()
    plt.pyplot.rc('axes', titlesize=font_axes)
    plt.pyplot.rc('axes', labelsize=font_axes)
    plt.pyplot.rc('xtick', labelsize=font_tick)
    plt.pyplot.rc('ytick', labelsize=font_tick)
    ax = fig.add_subplot(111)
    params = {'mathtext.default': 'regular'}
    ax.plot(x_range, y_range_ME)
    ax.scatter(x_range, y_range_ME)
    ax.set_ylabel('$log_{10}$ Condition number')
    ax.set_xlabel(xlabel_text)
    plt.pyplot.title(title_text + ' - Modified E optimality')
    plt.pyplot.show()