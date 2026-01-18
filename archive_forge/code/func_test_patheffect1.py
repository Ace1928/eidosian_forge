import numpy as np
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.path import Path
import matplotlib.patches as patches
@image_comparison(['patheffect1'], remove_text=True)
def test_patheffect1():
    ax1 = plt.subplot()
    ax1.imshow([[1, 2], [2, 3]])
    txt = ax1.annotate('test', (1.0, 1.0), (0.0, 0), arrowprops=dict(arrowstyle='->', connectionstyle='angle3', lw=2), size=20, ha='center', path_effects=[path_effects.withStroke(linewidth=3, foreground='w')])
    txt.arrow_patch.set_path_effects([path_effects.Stroke(linewidth=5, foreground='w'), path_effects.Normal()])
    pe = [path_effects.withStroke(linewidth=3, foreground='w')]
    ax1.grid(True, linestyle='-', path_effects=pe)