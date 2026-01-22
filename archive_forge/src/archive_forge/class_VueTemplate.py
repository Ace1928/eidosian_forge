import os
from traitlets import Any, Unicode, List, Dict, Union, Instance
from ipywidgets import DOMWidget
from ipywidgets.widgets.widget import widget_serialization
from .Template import Template, get_template
from ._version import semver
from .ForceLoad import force_load_instance
import inspect
from importlib import import_module
class VueTemplate(DOMWidget, Events):
    class_component_serialization = {'from_json': widget_serialization['to_json'], 'to_json': _class_to_json}
    _jupyter_vue = Any(force_load_instance, read_only=True).tag(sync=True, **widget_serialization)
    _model_name = Unicode('VueTemplateModel').tag(sync=True)
    _view_name = Unicode('VueView').tag(sync=True)
    _view_module = Unicode('jupyter-vue').tag(sync=True)
    _model_module = Unicode('jupyter-vue').tag(sync=True)
    _view_module_version = Unicode(semver).tag(sync=True)
    _model_module_version = Unicode(semver).tag(sync=True)
    template = Union([Instance(Template), Unicode()]).tag(sync=True, **widget_serialization)
    css = Unicode(None, allow_none=True).tag(sync=True)
    methods = Unicode(None, allow_none=True).tag(sync=True)
    data = Unicode(None, allow_none=True).tag(sync=True)
    events = List(Unicode(), allow_none=True).tag(sync=True)
    components = Dict(default_value=None, allow_none=True).tag(sync=True, **class_component_serialization)
    _component_instances = List().tag(sync=True, **widget_serialization)
    template_file = None

    def __init__(self, *args, **kwargs):
        if self.template_file:
            abs_path = ''
            if type(self.template_file) == str:
                abs_path = os.path.abspath(self.template_file)
            elif type(self.template_file) == tuple:
                rel_file, path = self.template_file
                abs_path = os.path.join(os.path.dirname(rel_file), path)
            self.template = get_template(abs_path)
        super().__init__(*args, **kwargs)
        sync_ref_traitlets = [v for k, v in self.traits().items() if 'sync_ref' in v.metadata.keys()]

        def create_ref_and_observe(traitlet):
            data = traitlet.get(self)
            ref_name = traitlet.name + '_ref'
            self.add_traits(**{ref_name: Any(as_refs(traitlet.name, data)).tag(sync=True)})

            def on_ref_source_change(change):
                setattr(self, ref_name, as_refs(traitlet.name, change['new']))
            self.observe(on_ref_source_change, traitlet.name)
        for traitlet in sync_ref_traitlets:
            create_ref_and_observe(traitlet)