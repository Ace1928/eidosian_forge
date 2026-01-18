from io import BytesIO
import numpy as np
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
def test_only_on_non_finite_bbox():
    fig, ax = plt.subplots()
    ax.annotate('', xy=(0, float('nan')))
    ax.set_axis_off()
    fig.savefig(BytesIO(), bbox_inches='tight', format='png')