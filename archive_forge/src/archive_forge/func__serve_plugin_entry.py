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
@wrappers.Request.application
def _serve_plugin_entry(self, request):
    """Serves a HTML for iframed plugin entry point.

        Args:
          request: The werkzeug.Request object.

        Returns:
          A werkzeug.Response object.
        """
    name = request.args.get('name')
    plugins = [plugin for plugin in self._plugins if plugin.plugin_name == name]
    if not plugins:
        raise errors.NotFoundError(name)
    if len(plugins) > 1:
        reason = 'Plugin invariant error: multiple plugins with name {name} found: {list}'.format(name=name, list=plugins)
        raise AssertionError(reason)
    plugin = plugins[0]
    module_path = plugin.frontend_metadata().es_module_path
    if not module_path:
        return http_util.Respond(request, 'Plugin is not module loadable', 'text/plain', code=400)
    if urlparse.urlparse(module_path).netloc:
        raise ValueError('Expected es_module_path to be non-absolute path')
    module_json = json.dumps('.' + module_path)
    script_content = 'import({}).then((m) => void m.render());'.format(module_json)
    digest = hashlib.sha256(script_content.encode('utf-8')).digest()
    script_sha = base64.b64encode(digest).decode('ascii')
    html = textwrap.dedent('\n            <!DOCTYPE html>\n            <head><base href="plugin/{name}/" /></head>\n            <body><script type="module">{script_content}</script></body>\n            ').format(name=name, script_content=script_content)
    return http_util.Respond(request, html, 'text/html', csp_scripts_sha256s=[script_sha])