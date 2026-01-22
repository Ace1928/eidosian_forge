from __future__ import annotations
import base64
import json
import sys
import zipfile
from abc import abstractmethod
from typing import (
from urllib.request import urlopen
import numpy as np
import param
from bokeh.models import LinearColorMapper
from bokeh.util.serialization import make_globally_unique_id
from pyviz_comms import JupyterComm
from ...param import ParamMethod
from ...util import isfile, lazy_load
from ..base import PaneBase
from ..plot import Bokeh
from .enums import PRESET_CMAPS
class BaseVTKRenderWindow(AbstractVTK):
    enable_keybindings = param.Boolean(default=False, doc='\n        Activate/Deactivate keys binding.\n\n        Warning: These keys bind may not work as expected in a notebook\n        context if they interact with already binded keys\n    ')
    serialize_on_instantiation = param.Boolean(default=False, constant=True, doc="\n         defines when the serialization of the vtkRenderWindow scene occurs.\n         If set to True the scene object is serialized when the pane is created\n         else (default) when the panel is displayed to the screen.\n\n         This parameter is constant, once set it can't be modified.\n\n         Warning: when the serialization occurs at instantiation, the vtkRenderWindow and\n         the view are not fully synchronized. The view displays the state of the scene\n         captured when the panel was created, if elements where added or removed between the\n         instantiation and the display these changes will not be reflected.\n         Moreover when the pane object is updated (replaced or call to param.trigger('object')),\n         all the scene is rebuilt from scratch.\n    ")
    serialize_all_data_arrays = param.Boolean(default=False, constant=True, doc='\n        If true, enable the serialization of all data arrays of vtkDataSets (point data, cell data and field data).\n        By default the value is False and only active scalars of each dataset are serialized and transfer to the\n        javascript side.\n\n        Enabling this option will increase memory and network transfer volume but results in more reactive visualizations\n        by using some custom javascript functions.\n    ')
    _applies_kw = True
    _rename: ClassVar[Mapping[str, str | None]] = {'serialize_on_instantiation': None, 'serialize_all_data_arrays': None}
    __abstract = True

    def __init__(self, object, **params):
        self._debug_serializer = params.pop('debug_serializer', False)
        super().__init__(object, **params)
        import panel.pane.vtk.synchronizable_serializer as rws
        rws.initializeSerializers()

    @classmethod
    def applies(cls, obj, **kwargs):
        if 'vtk' not in sys.modules and 'vtkmodules' not in sys.modules:
            return False
        else:
            import vtk
            return isinstance(obj, vtk.vtkRenderWindow)

    def get_renderer(self):
        """
        Get the vtk Renderer associated to this pane
        """
        return list(self.object.GetRenderers())[0]

    def _vtklut2bkcmap(self, lut, name):
        table = lut.GetTable()
        low, high = lut.GetTableRange()
        rgba_arr = np.frombuffer(memoryview(table), dtype=np.uint8).reshape((-1, 4))
        palette = [self._rgb2hex(*rgb) for rgb in rgba_arr[:, :3]]
        return LinearColorMapper(low=low, high=high, name=name, palette=palette)

    def get_color_mappers(self, infer=False):
        if not infer:
            cmaps = []
            for view_prop in self.get_renderer().GetViewProps():
                if view_prop.IsA('vtkScalarBarActor'):
                    name = view_prop.GetTitle()
                    lut = view_prop.GetLookupTable()
                    cmaps.append(self._vtklut2bkcmap(lut, name))
        else:
            infered_cmaps = {}
            for actor in self.get_renderer().GetActors():
                mapper = actor.GetMapper()
                cmap_name = mapper.GetArrayName()
                if cmap_name and cmap_name not in infered_cmaps:
                    lut = mapper.GetLookupTable()
                    infered_cmaps[cmap_name] = self._vtklut2bkcmap(lut, cmap_name)
            cmaps = infered_cmaps.values()
        return cmaps

    @param.depends('color_mappers')
    def _construct_colorbars(self, color_mappers=None):
        if not color_mappers:
            color_mappers = self.color_mappers
        from bokeh.models import ColorBar, FixedTicker, Plot
        cbs = []
        for color_mapper in color_mappers:
            ticks = np.linspace(color_mapper.low, color_mapper.high, 5)
            cbs.append(ColorBar(color_mapper=color_mapper, title=color_mapper.name, ticker=FixedTicker(ticks=ticks), label_standoff=5, background_fill_alpha=0, orientation='horizontal', location=(0, 0)))
        plot = Plot(toolbar_location=None, frame_height=0, sizing_mode='stretch_width', outline_line_width=0)
        [plot.add_layout(cb, 'below') for cb in cbs]
        return plot

    def construct_colorbars(self, infer=True):
        if infer:
            color_mappers = self.get_color_mappers(infer=True)
            model = self._construct_colorbars(color_mappers)
            return Bokeh(model)
        else:
            return ParamMethod(self._construct_colorbars)

    def export_scene(self, filename='vtk_scene', all_data_arrays=False):
        if '.' not in filename:
            filename += '.synch'
        import panel.pane.vtk.synchronizable_serializer as rws
        context = rws.SynchronizationContext(serialize_all_data_arrays=all_data_arrays, debug=self._debug_serializer)
        scene, arrays, annotations = self._serialize_ren_win(self.object, context, binary=True, compression=False)
        with zipfile.ZipFile(filename, mode='w') as zf:
            zf.writestr('index.json', json.dumps(scene))
            for name, data in arrays.items():
                zf.writestr('data/%s' % name, data, zipfile.ZIP_DEFLATED)
            zf.writestr('annotations.json', json.dumps(annotations))
        return filename

    def _update_color_mappers(self):
        color_mappers = self.get_color_mappers()
        if self.color_mappers != color_mappers:
            self.color_mappers = color_mappers

    def _serialize_ren_win(self, ren_win, context, binary=False, compression=True, exclude_arrays=None):
        import panel.pane.vtk.synchronizable_serializer as rws
        if exclude_arrays is None:
            exclude_arrays = []
        ren_win.OffScreenRenderingOn()
        ren_win.Modified()
        ren_win.Render()
        scene = rws.serializeInstance(None, ren_win, context.getReferenceId(ren_win), context, 0)
        scene['properties']['numberOfLayers'] = 2
        arrays = {name: context.getCachedDataArray(name, binary=True, compression=False) for name in context.dataArrayCache.keys() if name not in exclude_arrays}
        annotations = context.getAnnotations()
        return (scene, arrays, annotations)

    @staticmethod
    def _rgb2hex(r, g, b):
        int_type = (int, np.integer)
        if isinstance(r, int_type) and isinstance(g, int_type) is isinstance(b, int_type):
            return '#{0:02x}{1:02x}{2:02x}'.format(r, g, b)
        else:
            return '#{0:02x}{1:02x}{2:02x}'.format(int(255 * r), int(255 * g), int(255 * b))