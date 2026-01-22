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
def _get_norm_opts(self, obj):
    """
        Gets the normalization options for a LabelledData object by
        traversing the object to find elements and their ids.
        The id is then used to select the appropriate OptionsTree,
        accumulating the normalization options into a dictionary.
        Returns a dictionary of normalization options for each
        element in the tree.
        """
    norm_opts = {}
    type_val_fn = lambda x: (x.id, (type(x).__name__, util.group_sanitizer(x.group, escape=False), util.label_sanitizer(x.label, escape=False))) if isinstance(x, Element) else None
    element_specs = {(idspec[0], idspec[1]) for idspec in obj.traverse(type_val_fn) if idspec is not None}
    key_fn = lambda x: -1 if x[0] is None else x[0]
    id_groups = groupby(sorted(element_specs, key=key_fn), key_fn)
    for gid, element_spec_group in id_groups:
        gid = None if gid == -1 else gid
        group_specs = [el for _, el in element_spec_group]
        backend = self.renderer.backend
        optstree = Store.custom_options(backend=backend).get(gid, Store.options(backend=backend))
        for opts in optstree:
            path = tuple(opts.path.split('.')[1:])
            applies = any((path == spec[:i] for spec in group_specs for i in range(1, 4)))
            if applies and 'norm' in opts.groups:
                nopts = opts['norm'].options
                popts = opts['plot'].options
                if 'axiswise' in nopts or 'framewise' in nopts or 'clim_percentile' in popts:
                    norm_opts.update({path: (nopts.get('axiswise', False), nopts.get('framewise', False), popts.get('clim_percentile', False))})
    element_specs = [spec for _, spec in element_specs]
    norm_opts.update({spec: (False, False, False) for spec in element_specs if not any((spec[:i] in norm_opts.keys() for i in range(1, 4)))})
    return norm_opts