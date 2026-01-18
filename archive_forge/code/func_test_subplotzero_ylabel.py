import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
from matplotlib.transforms import IdentityTransform
from mpl_toolkits.axisartist.axislines import AxesZero, SubplotZero, Subplot
from mpl_toolkits.axisartist import Axes, SubplotHost
@image_comparison(['subplotzero_ylabel.png'], style='mpl20')
def test_subplotzero_ylabel():
    fig = plt.figure()
    ax = fig.add_subplot(111, axes_class=SubplotZero)
    ax.set(xlim=(-3, 7), ylim=(-3, 7), xlabel='x', ylabel='y')
    zero_axis = ax.axis['xzero', 'yzero']
    zero_axis.set_visible(True)
    ax.axis['left', 'right', 'bottom', 'top'].set_visible(False)
    zero_axis.set_axisline_style('->')