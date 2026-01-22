import numpy as np
import matplotlib as mpl
from matplotlib import _api, cm, patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
class StreamplotSet:

    def __init__(self, lines, arrows):
        self.lines = lines
        self.arrows = arrows