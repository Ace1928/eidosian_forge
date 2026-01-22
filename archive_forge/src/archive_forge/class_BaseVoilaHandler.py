import asyncio
import os
from typing import Dict
from pathlib import Path
import tornado.web
from jupyter_server.base.handlers import JupyterHandler
from jupyter_server.utils import url_path_join
from nbclient.util import ensure_async
from tornado.httputil import split_host_and_port
from traitlets.traitlets import Bool
from ._version import __version__
from .notebook_renderer import NotebookRenderer
from .request_info_handler import RequestInfoSocketHandler
from .utils import ENV_VARIABLE, create_include_assets_functions
class BaseVoilaHandler(JupyterHandler):

    def initialize(self, **kwargs):
        self.voila_configuration = kwargs['voila_configuration']

    def render_template(self, name, **ns):
        """ Render the Voila HTML template, respecting the theme and nbconvert template.
        """
        template_arg = self.get_argument('voila-template', self.voila_configuration.template) if self.voila_configuration.allow_template_override == 'YES' else self.voila_configuration.template
        theme_arg = self.get_argument('voila-theme', self.voila_configuration.theme) if self.voila_configuration.allow_theme_override == 'YES' else self.voila_configuration.theme
        ns = {**ns, **self.template_namespace, **create_include_assets_functions(template_arg, self.base_url), 'theme': theme_arg}
        template = self.get_template(name)
        return template.render(**ns)

    def get_template(self, name):
        """Return the jinja template object for a given name"""
        voila_env = self.settings['voila_jinja2_env']
        template = voila_env.get_template(name)
        if template is not None:
            return template
        return self.settings['jinja2_env'].get_template(name)