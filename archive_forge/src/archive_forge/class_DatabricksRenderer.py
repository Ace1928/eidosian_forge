import base64
import json
import webbrowser
import inspect
import os
from os.path import isdir
from plotly import utils, optional_imports
from plotly.io import to_json, to_image, write_image, write_html
from plotly.io._orca import ensure_server
from plotly.io._utils import plotly_cdn_url
from plotly.offline.offline import _get_jconfig, get_plotlyjs
from plotly.tools import return_figure_from_figure_or_data
class DatabricksRenderer(ExternalRenderer):

    def __init__(self, config=None, auto_play=False, post_script=None, animation_opts=None, include_plotlyjs='cdn'):
        self.config = config
        self.auto_play = auto_play
        self.post_script = post_script
        self.animation_opts = animation_opts
        self.include_plotlyjs = include_plotlyjs
        self._displayHTML = None

    @property
    def displayHTML(self):
        import inspect
        if self._displayHTML is None:
            for frame in inspect.getouterframes(inspect.currentframe()):
                global_names = set(frame.frame.f_globals)
                if all((v in global_names for v in ['displayHTML', 'display', 'spark'])):
                    self._displayHTML = frame.frame.f_globals['displayHTML']
                    break
            if self._displayHTML is None:
                raise EnvironmentError("\nUnable to detect the Databricks displayHTML function. The 'databricks' renderer is only\nsupported when called from within the Databricks notebook environment.")
        return self._displayHTML

    def render(self, fig_dict):
        from plotly.io import to_html
        html = to_html(fig_dict, config=self.config, auto_play=self.auto_play, include_plotlyjs=self.include_plotlyjs, include_mathjax='cdn', post_script=self.post_script, full_html=True, animation_opts=self.animation_opts, default_width='100%', default_height='100%', validate=False)
        self.displayHTML(html)