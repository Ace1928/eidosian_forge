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
class JsonRenderer(MimetypeRenderer):
    """
    Renderer to display figures as JSON hierarchies.  This renderer is
    compatible with JupyterLab and VSCode.

    mime type: 'application/json'
    """

    def to_mimebundle(self, fig_dict):
        value = json.loads(to_json(fig_dict, validate=False, remove_uids=False))
        return {'application/json': value}