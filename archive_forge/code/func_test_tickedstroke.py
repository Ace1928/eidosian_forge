import numpy as np
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.path import Path
import matplotlib.patches as patches
@image_comparison(['tickedstroke'], remove_text=True, extensions=['png'], tol=0.22)
def test_tickedstroke():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    path = Path.unit_circle()
    patch = patches.PathPatch(path, facecolor='none', lw=2, path_effects=[path_effects.withTickedStroke(angle=-90, spacing=10, length=1)])
    ax1.add_patch(patch)
    ax1.axis('equal')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax2.plot([0, 1], [0, 1], label=' ', path_effects=[path_effects.withTickedStroke(spacing=7, angle=135)])
    nx = 101
    x = np.linspace(0.0, 1.0, nx)
    y = 0.3 * np.sin(x * 8) + 0.4
    ax2.plot(x, y, label=' ', path_effects=[path_effects.withTickedStroke()])
    ax2.legend()
    nx = 101
    ny = 105
    xvec = np.linspace(0.001, 4.0, nx)
    yvec = np.linspace(0.001, 4.0, ny)
    x1, x2 = np.meshgrid(xvec, yvec)
    g1 = -(3 * x1 + x2 - 5.5)
    g2 = -(x1 + 2 * x2 - 4)
    g3 = 0.8 + x1 ** (-3) - x2
    cg1 = ax3.contour(x1, x2, g1, [0], colors=('k',))
    cg1.set(path_effects=[path_effects.withTickedStroke(angle=135)])
    cg2 = ax3.contour(x1, x2, g2, [0], colors=('r',))
    cg2.set(path_effects=[path_effects.withTickedStroke(angle=60, length=2)])
    cg3 = ax3.contour(x1, x2, g3, [0], colors=('b',))
    cg3.set(path_effects=[path_effects.withTickedStroke(spacing=7)])
    ax3.set_xlim(0, 4)
    ax3.set_ylim(0, 4)