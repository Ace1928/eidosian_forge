import uuid
import warnings
from ast import literal_eval
from collections import Counter, defaultdict
from functools import partial
from itertools import groupby, product
import numpy as np
import param
from panel.config import config
from panel.io.document import unlocked
from panel.io.notebook import push
from panel.io.state import state
from pyviz_comms import JupyterComm
from ..core import traversal, util
from ..core.data import Dataset, disable_pipeline
from ..core.element import Element, Element3D
from ..core.layout import Empty, Layout, NdLayout
from ..core.options import Compositor, SkipRendering, Store, lookup_options
from ..core.overlay import CompositeOverlay, NdOverlay, Overlay
from ..core.spaces import DynamicMap, HoloMap
from ..core.util import isfinite, stream_parameters
from ..element import Graph, Table
from ..selection import NoOpSelectionDisplay
from ..streams import RangeX, RangeXY, RangeY, Stream
from ..util.transform import dim
from .util import (
def _define_interface(self, plots, allow_mismatch):
    parameters = [{k: v.precedence for k, v in plot.param.objects().items() if v.precedence is None or v.precedence >= 0} for plot in plots]
    param_sets = [set(params.keys()) for params in parameters]
    if not allow_mismatch and (not all((pset == param_sets[0] for pset in param_sets))):
        mismatching_sets = [pset for pset in param_sets if pset != param_sets[0]]
        for mismatch_set in mismatching_sets:
            print('Mismatching plot options:', mismatch_set)
        raise Exception('All selectable plot classes must have identical plot options.')
    styles = [plot.style_opts for plot in plots]
    if not allow_mismatch and (not all((style == styles[0] for style in styles))):
        raise Exception('All selectable plot classes must have identical style options.')
    plot_params = {p: v for params in parameters for p, v in params.items()}
    return ([s for style in styles for s in style], plot_params)