from __future__ import absolute_import
import numpy as np
import ipywidgets as widgets  # we should not have widgets under two names
import traitlets
from traitlets import Unicode
from traittypes import Array
import ipyvolume._version
from ipyvolume import serialize
class TransferFunctionJsBumps(TransferFunction):
    _model_name = Unicode('TransferFunctionJsBumpsModel').tag(sync=True)
    _model_module = Unicode('ipyvolume').tag(sync=True)
    levels = traitlets.List(traitlets.CFloat(), default_value=[0.1, 0.5, 0.8]).tag(sync=True)
    opacities = traitlets.List(traitlets.CFloat(), default_value=[0.01, 0.05, 0.1]).tag(sync=True)
    widths = traitlets.List(traitlets.CFloat(), default_value=[0.1, 0.1, 0.1]).tag(sync=True)

    def control(self, max_opacity=0.2):
        return widgets.VBox()