import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D, Transform
from matplotlib.testing.decorators import image_comparison
from mpl_toolkits.axisartist import SubplotHost
from mpl_toolkits.axes_grid1.parasite_axes import host_axes_class_factory
from mpl_toolkits.axisartist import angle_helper
from mpl_toolkits.axisartist.axislines import Axes
from mpl_toolkits.axisartist.grid_helper_curvelinear import \
@image_comparison(['polar_box.png'], style='default', tol=0.27)
def test_polar_box():
    plt.rcParams['text.kerning_factor'] = 6
    fig = plt.figure(figsize=(5, 5))
    tr = Affine2D().scale(np.pi / 180.0, 1.0) + PolarAxes.PolarTransform()
    extreme_finder = angle_helper.ExtremeFinderCycle(20, 20, lon_cycle=360, lat_cycle=None, lon_minmax=None, lat_minmax=(0, np.inf))
    grid_locator1 = angle_helper.LocatorDMS(12)
    tick_formatter1 = angle_helper.FormatterDMS()
    grid_helper = GridHelperCurveLinear(tr, extreme_finder=extreme_finder, grid_locator1=grid_locator1, tick_formatter1=tick_formatter1)
    ax1 = SubplotHost(fig, 1, 1, 1, grid_helper=grid_helper)
    ax1.axis['right'].major_ticklabels.set_visible(True)
    ax1.axis['top'].major_ticklabels.set_visible(True)
    ax1.axis['right'].get_helper().nth_coord_ticks = 0
    ax1.axis['bottom'].get_helper().nth_coord_ticks = 1
    fig.add_subplot(ax1)
    ax1.axis['lat'] = axis = grid_helper.new_floating_axis(0, 45, axes=ax1)
    axis.label.set_text('Test')
    axis.label.set_visible(True)
    axis.get_helper().set_extremes(2, 12)
    ax1.axis['lon'] = axis = grid_helper.new_floating_axis(1, 6, axes=ax1)
    axis.label.set_text('Test 2')
    axis.get_helper().set_extremes(-180, 90)
    ax2 = ax1.get_aux_axes(tr, viewlim_mode='equal')
    assert ax2.transData == tr + ax1.transData
    ax2.plot(np.linspace(0, 30, 50), np.linspace(10, 10, 50))
    ax1.set_aspect(1.0)
    ax1.set_xlim(-5, 12)
    ax1.set_ylim(-5, 10)
    ax1.grid(True)