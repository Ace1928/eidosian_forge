import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
from mpl_toolkits.axisartist import AxisArtistHelperRectlinear
from mpl_toolkits.axisartist.axis_artist import (AxisArtist, AxisLabel,
@image_comparison(['axis_artist_ticks.png'], style='default')
def test_ticks():
    fig, ax = plt.subplots()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    locs_angles = [((i / 10, 0.0), i * 30) for i in range(-1, 12)]
    ticks_in = Ticks(ticksize=10, axis=ax.xaxis)
    ticks_in.set_locs_angles(locs_angles)
    ax.add_artist(ticks_in)
    ticks_out = Ticks(ticksize=10, tick_out=True, color='C3', axis=ax.xaxis)
    ticks_out.set_locs_angles(locs_angles)
    ax.add_artist(ticks_out)