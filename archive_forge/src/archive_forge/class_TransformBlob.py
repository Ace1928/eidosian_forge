from io import BytesIO
import ast
import pickle
import pickletools
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import cm
from matplotlib.testing import subprocess_run_helper
from matplotlib.testing.decorators import check_figures_equal
from matplotlib.dates import rrulewrapper
from matplotlib.lines import VertexSelector
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.figure as mfigure
from mpl_toolkits.axes_grid1 import parasite_axes  # type: ignore
class TransformBlob:

    def __init__(self):
        self.identity = mtransforms.IdentityTransform()
        self.identity2 = mtransforms.IdentityTransform()
        self.composite = mtransforms.CompositeGenericTransform(self.identity, self.identity2)
        self.wrapper = mtransforms.TransformWrapper(self.composite)
        self.composite2 = mtransforms.CompositeGenericTransform(self.wrapper, self.identity)