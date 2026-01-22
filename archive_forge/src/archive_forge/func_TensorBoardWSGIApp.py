import base64
import collections
import hashlib
import io
import json
import re
import textwrap
import time
from urllib import parse as urlparse
import zipfile
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import auth_context_middleware
from tensorboard.backend import client_feature_flags
from tensorboard.backend import empty_path_redirect
from tensorboard.backend import experiment_id
from tensorboard.backend import experimental_plugin
from tensorboard.backend import http_util
from tensorboard.backend import path_prefix
from tensorboard.backend import security_validator
from tensorboard.plugins import base_plugin
from tensorboard.plugins.core import core_plugin
from tensorboard.util import tb_logging
def TensorBoardWSGIApp(flags, plugins, data_provider=None, assets_zip_provider=None, deprecated_multiplexer=None, auth_providers=None, experimental_middlewares=None):
    """Constructs a TensorBoard WSGI app from plugins and data providers.

    Args:
      flags: An argparse.Namespace containing TensorBoard CLI flags.
      plugins: A list of plugins, which can be provided as TBPlugin subclasses
          or TBLoader instances or subclasses.
      data_provider: Instance of `tensorboard.data.provider.DataProvider`. May
          be `None` if `flags.generic_data` is set to `"false"` in which case
          `deprecated_multiplexer` must be passed instead.
      assets_zip_provider: See TBContext documentation for more information. If
          `None` a placeholder assets zipfile will be used containing only a
          default `index.html` file, and the actual frontend assets must be
          supplied by middleware wrapping this WSGI app.
      deprecated_multiplexer: Optional `plugin_event_multiplexer.EventMultiplexer`
          to use for any plugins not yet enabled for the DataProvider API.
          Required if the data_provider argument is not passed.
      auth_providers: Optional mapping whose values are `AuthProvider` values
        and whose keys are used by (e.g.) data providers to specify
        `AuthProvider`s via the `AuthContext.get` interface. Defaults to `{}`.
      experimental_middlewares: Optional list of WSGI middlewares (i.e.,
        callables that take a WSGI application and return a WSGI application)
        to apply directly around the core TensorBoard app itself, "inside" the
        request redirection machinery for `--path_prefix`, experiment IDs, etc.
        You can use this to add handlers for additional routes. Middlewares are
        applied in listed order, so the first element of this list is the
        innermost application. Defaults to `[]`. This parameter is experimental
        and may be reworked or removed.

    Returns:
      A WSGI application that implements the TensorBoard backend.

    :type plugins: list[base_plugin.TBLoader]
    """
    if assets_zip_provider is None:
        assets_zip_provider = _placeholder_assets_zip_provider
    plugin_name_to_instance = {}
    context = base_plugin.TBContext(data_provider=data_provider, flags=flags, logdir=flags.logdir, multiplexer=deprecated_multiplexer, assets_zip_provider=assets_zip_provider, plugin_name_to_instance=plugin_name_to_instance, sampling_hints=flags.samples_per_plugin, window_title=flags.window_title)
    tbplugins = []
    experimental_plugins = []
    for plugin_spec in plugins:
        loader = make_plugin_loader(plugin_spec)
        try:
            plugin = loader.load(context)
        except Exception:
            logger.error('Failed to load plugin %s; ignoring it.', getattr(loader.load, '__qualname__', loader.load), exc_info=True)
            plugin = None
        if plugin is None:
            continue
        tbplugins.append(plugin)
        if isinstance(loader, experimental_plugin.ExperimentalPlugin) or isinstance(plugin, experimental_plugin.ExperimentalPlugin):
            experimental_plugins.append(plugin.plugin_name)
        plugin_name_to_instance[plugin.plugin_name] = plugin
    return TensorBoardWSGI(tbplugins, flags.path_prefix, data_provider, experimental_plugins, auth_providers, experimental_middlewares)