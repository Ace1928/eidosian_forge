from io import BytesIO
import numpy as np
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
def y_formatter(y, pos):
    if int(y) == 4:
        return 'The number 4'
    else:
        return str(y)