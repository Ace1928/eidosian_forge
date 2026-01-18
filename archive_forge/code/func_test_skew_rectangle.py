from contextlib import ExitStack
import itertools
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
from matplotlib.axes import Axes
import matplotlib.transforms as transforms
import matplotlib.axis as maxis
import matplotlib.spines as mspines
import matplotlib.patches as mpatch
from matplotlib.projections import register_projection
@image_comparison(['skew_rects'], remove_text=True)
def test_skew_rectangle():
    fix, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(8, 8))
    axes = axes.flat
    rotations = list(itertools.product([-3, -1, 0, 1, 3], repeat=2))
    axes[0].set_xlim([-3, 3])
    axes[0].set_ylim([-3, 3])
    axes[0].set_aspect('equal', share=True)
    for ax, (xrots, yrots) in zip(axes, rotations):
        xdeg, ydeg = (45 * xrots, 45 * yrots)
        t = transforms.Affine2D().skew_deg(xdeg, ydeg)
        ax.set_title(f'Skew of {xdeg} in X and {ydeg} in Y')
        ax.add_patch(mpatch.Rectangle([-1, -1], 2, 2, transform=t + ax.transData, alpha=0.5, facecolor='coral'))
    plt.subplots_adjust(wspace=0, left=0.01, right=0.99, bottom=0.01, top=0.99)