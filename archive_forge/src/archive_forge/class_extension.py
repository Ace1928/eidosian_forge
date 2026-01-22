import inspect
import os
import shutil
import sys
from collections import defaultdict
from inspect import Parameter, Signature
from pathlib import Path
from types import FunctionType
import param
from pyviz_comms import extension as _pyviz_extension
from ..core import (
from ..core.operation import Operation, OperationCallable
from ..core.options import Keywords, Options, options_policy
from ..core.overlay import Overlay
from ..core.util import merge_options_to_dict
from ..operation.element import function
from ..streams import Params, Stream, streams_list_from_dict
from .settings import OutputSettings, list_backends, list_formats
class extension(_pyviz_extension):
    """
    Helper utility used to load holoviews extensions. These can be
    plotting extensions, element extensions or anything else that can be
    registered to work with HoloViews.
    """
    _backends = {'matplotlib': 'mpl', 'bokeh': 'bokeh', 'plotly': 'plotly'}
    _backend_hooks = defaultdict(list)
    _loaded = False

    def __call__(self, *args, **params):
        config = params.pop('config', {})
        util.config.param.update(**config)
        imports = [(arg, self._backends[arg]) for arg in args if arg in self._backends]
        for p, _val in sorted(params.items()):
            if p in self._backends:
                imports.append((p, self._backends[p]))
        if not imports:
            args = ['matplotlib']
            imports = [('matplotlib', 'mpl')]
        args = list(args)
        selected_backend = None
        for backend, imp in imports:
            try:
                __import__(backend)
            except ImportError:
                self.param.warning(f'{backend} could not be imported, ensure {backend} is installed.')
            try:
                __import__(f'holoviews.plotting.{imp}')
                if selected_backend is None:
                    selected_backend = backend
            except util.VersionError as e:
                self.param.warning(f'HoloViews {backend} extension could not be loaded. The installed {backend} version {e.version} is less than the required version {e.min_version}.')
            except Exception as e:
                self.param.warning(f"Holoviews {backend} extension could not be imported, it raised the following exception: {type(e).__name__}('{e}')")
            finally:
                Store.output_settings.allowed['backend'] = list_backends()
                Store.output_settings.allowed['fig'] = list_formats('fig', backend)
                Store.output_settings.allowed['holomap'] = list_formats('holomap', backend)
            for hook in self._backend_hooks[backend]:
                try:
                    hook()
                except Exception as e:
                    self.param.warning(f'{backend} backend hook {hook} failed with following exception: {e}')
        if selected_backend is None:
            raise ImportError('None of the backends could be imported')
        Store.set_current_backend(selected_backend)
        import panel as pn
        if params.get('enable_mathjax', False) and selected_backend == 'bokeh':
            pn.extension('mathjax')
        if pn.config.comms == 'default':
            if 'google.colab' in sys.modules:
                pn.config.comms = 'colab'
                return
            if 'VSCODE_CWD' in os.environ or 'VSCODE_PID' in os.environ:
                pn.config.comms = 'vscode'
                self._ignore_bokeh_warnings()
                return

    @classmethod
    def register_backend_callback(cls, backend, callback):
        """Registers a hook which is run when a backend is loaded"""
        cls._backend_hooks[backend].append(callback)

    def _ignore_bokeh_warnings(self):
        import warnings
        from bokeh.util.warnings import BokehUserWarning
        warnings.filterwarnings('ignore', category=BokehUserWarning, message='reference already known')