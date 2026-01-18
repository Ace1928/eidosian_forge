from itertools import cycle
import numpy as np
from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
def update_from_first_child(tgt, src):
    first_child = next(iter(src.get_children()), None)
    if first_child is not None:
        tgt.update_from(first_child)