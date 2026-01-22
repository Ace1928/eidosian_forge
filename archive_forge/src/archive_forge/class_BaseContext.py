import atexit
import faulthandler
import functools
import inspect
import io
import json
import logging
import os
import sys
import threading
import time
import traceback
import urllib
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
from urllib.parse import urlparse
import colorama
import setproctitle
from typing import Literal, Protocol
import ray
import ray._private.node
import ray._private.parameter
import ray._private.profiling as profiling
import ray._private.ray_constants as ray_constants
import ray._private.serialization as serialization
import ray._private.services as services
import ray._private.state
import ray._private.storage as storage
import ray.actor
import ray.cloudpickle as pickle  # noqa
import ray.job_config
import ray.remote_function
from ray import ActorID, JobID, Language, ObjectRef
from ray._raylet import raise_sys_exit_with_custom_error_message
from ray._raylet import ObjectRefGenerator, TaskID
from ray.runtime_env.runtime_env import _merge_runtime_env
from ray._private import ray_option_utils
from ray._private.client_mode_hook import client_mode_hook
from ray._private.function_manager import FunctionActorManager
from ray._private.inspect_util import is_cython
from ray._private.ray_logging import (
from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR
from ray._private.runtime_env.py_modules import upload_py_modules_if_needed
from ray._private.runtime_env.working_dir import upload_working_dir_if_needed
from ray._private.runtime_env.setup_hook import (
from ray._private.storage import _load_class
from ray._private.utils import get_ray_doc_version
from ray.exceptions import ObjectStoreFullError, RayError, RaySystemError, RayTaskError
from ray.experimental.internal_kv import (
from ray.experimental import tqdm_ray
from ray.experimental.tqdm_ray import RAY_TQDM_MAGIC
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
from ray.util.debug import log_once
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.tracing.tracing_helper import _import_from_string
from ray.widgets import Template
from ray.widgets.util import repr_with_fallback
class BaseContext(metaclass=ABCMeta):
    """
    Base class for RayContext and ClientContext
    """
    dashboard_url: Optional[str]
    python_version: str
    ray_version: str

    @abstractmethod
    def disconnect(self):
        """
        If this context is for directly attaching to a cluster, disconnect
        will call ray.shutdown(). Otherwise, if the context is for a ray
        client connection, the client will be disconnected.
        """
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self):
        pass

    def _context_table_template(self):
        if self.dashboard_url:
            dashboard_row = Template('context_dashrow.html.j2').render(dashboard_url='http://' + self.dashboard_url)
        else:
            dashboard_row = None
        return Template('context_table.html.j2').render(python_version=self.python_version, ray_version=self.ray_version, dashboard_row=dashboard_row)

    def _repr_html_(self):
        return Template('context.html.j2').render(context_logo=Template('context_logo.html.j2').render(), context_table=self._context_table_template())

    @repr_with_fallback(['ipywidgets', '8'])
    def _get_widget_bundle(self, **kwargs) -> Dict[str, Any]:
        """Get the mimebundle for the widget representation of the context.

        Args:
            **kwargs: Passed to the _repr_mimebundle_() function for the widget

        Returns:
            Dictionary ("mimebundle") of the widget representation of the context.
        """
        import ipywidgets
        disconnect_button = ipywidgets.Button(description='Disconnect', disabled=False, button_style='', tooltip='Disconnect from the Ray cluster', layout=ipywidgets.Layout(margin='auto 0px 0px 0px'))

        def disconnect_callback(button):
            button.disabled = True
            button.description = 'Disconnecting...'
            self.disconnect()
            button.description = 'Disconnected'
        disconnect_button.on_click(disconnect_callback)
        left_content = ipywidgets.VBox([ipywidgets.HTML(Template('context_logo.html.j2').render()), disconnect_button], layout=ipywidgets.Layout())
        right_content = ipywidgets.HTML(self._context_table_template())
        widget = ipywidgets.HBox([left_content, right_content], layout=ipywidgets.Layout(width='100%'))
        return widget._repr_mimebundle_(**kwargs)

    def _repr_mimebundle_(self, **kwargs):
        bundle = self._get_widget_bundle(**kwargs)
        bundle.update({'text/html': self._repr_html_(), 'text/plain': repr(self)})
        return bundle