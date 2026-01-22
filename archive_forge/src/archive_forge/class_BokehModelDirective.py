from __future__ import annotations
import logging  # isort:skip
import importlib
import json
import warnings
from os import getenv
from typing import Any
from docutils.parsers.rst.directives import unchanged
from sphinx.errors import SphinxError
from bokeh.core.property.singletons import Undefined
from bokeh.core.serialization import AnyRep, Serializer, SymbolRep
from bokeh.model import Model
from bokeh.util.warnings import BokehDeprecationWarning
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective, py_sig_re
from .templates import MODEL_DETAIL
class BokehModelDirective(BokehDirective):
    has_content = True
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'module': unchanged, 'canonical': unchanged}

    def run(self):
        sig = ' '.join(self.arguments)
        m = py_sig_re.match(sig)
        if m is None:
            raise SphinxError(f'Unable to parse signature for bokeh-model: {sig!r}')
        name_prefix, model_name, arglist, retann = m.groups()
        if getenv('BOKEH_SPHINX_QUICK') == '1':
            return self.parse(f'{model_name}\n{'-' * len(model_name)}\n', '<bokeh-model>')
        module_name = self.options['module']
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            raise SphinxError(f"Unable to generate model reference docs for {model_name}, couldn't import module {module_name}")
        model = getattr(module, model_name, None)
        if model is None:
            raise SphinxError(f'Unable to generate model reference docs: no model for {model_name} in module {module_name}')
        if not issubclass(model, Model):
            raise SphinxError(f'Unable to generate model reference docs: {model_name}, is not a subclass of Model')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=BokehDeprecationWarning)
            model_obj = model()
        model_json = json.dumps(to_json_rep(model_obj), sort_keys=True, indent=2, separators=(', ', ': '))
        adjusted_module_name = 'bokeh.models' if module_name.startswith('bokeh.models') else module_name
        rst_text = MODEL_DETAIL.render(name=model_name, module_name=adjusted_module_name, model_json=model_json)
        return self.parse(rst_text, '<bokeh-model>')