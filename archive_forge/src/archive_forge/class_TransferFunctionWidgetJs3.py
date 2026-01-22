from __future__ import absolute_import
import numpy as np
import ipywidgets as widgets  # we should not have widgets under two names
import traitlets
from traitlets import Unicode
from traittypes import Array
import ipyvolume._version
from ipyvolume import serialize
class TransferFunctionWidgetJs3(TransferFunction):
    _model_name = Unicode('TransferFunctionWidgetJs3Model').tag(sync=True)
    _model_module = Unicode('ipyvolume').tag(sync=True)
    level1 = traitlets.Float(0.1).tag(sync=True)
    level2 = traitlets.Float(0.5).tag(sync=True)
    level3 = traitlets.Float(0.8).tag(sync=True)
    opacity1 = traitlets.Float(0.01).tag(sync=True)
    opacity2 = traitlets.Float(0.05).tag(sync=True)
    opacity3 = traitlets.Float(0.1).tag(sync=True)
    width1 = traitlets.Float(0.1).tag(sync=True)
    width2 = traitlets.Float(0.1).tag(sync=True)
    width3 = traitlets.Float(0.1).tag(sync=True)

    def control(self, max_opacity=0.2):
        l1 = widgets.FloatSlider(min=0, max=1, step=0.001, value=self.level1)
        l2 = widgets.FloatSlider(min=0, max=1, step=0.001, value=self.level2)
        l3 = widgets.FloatSlider(min=0, max=1, step=0.001, value=self.level3)
        o1 = widgets.FloatSlider(min=0, max=max_opacity, step=0.001, value=self.opacity1)
        o2 = widgets.FloatSlider(min=0, max=max_opacity, step=0.001, value=self.opacity2)
        o3 = widgets.FloatSlider(min=0, max=max_opacity, step=0.001, value=self.opacity2)
        widgets.jslink((self, 'level1'), (l1, 'value'))
        widgets.jslink((self, 'level2'), (l2, 'value'))
        widgets.jslink((self, 'level3'), (l3, 'value'))
        widgets.jslink((self, 'opacity1'), (o1, 'value'))
        widgets.jslink((self, 'opacity2'), (o2, 'value'))
        widgets.jslink((self, 'opacity3'), (o3, 'value'))
        return widgets.VBox([widgets.HBox([widgets.Label(value='levels:'), l1, l2, l3]), widgets.HBox([widgets.Label(value='opacities:'), o1, o2, o3])])