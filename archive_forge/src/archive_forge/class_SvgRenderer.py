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
class SvgRenderer(ImageRenderer):
    """
    Renderer to display figures as static SVG images.  This renderer requires
    either the kaleido package or the orca command-line utility and is broadly
    compatible across IPython environments (classic Jupyter Notebook, JupyterLab,
    QtConsole, VSCode, PyCharm, etc) and nbconvert targets (HTML, PDF, etc.).

    mime type: 'image/svg+xml'
    """

    def __init__(self, width=None, height=None, scale=None, engine='auto'):
        super(SvgRenderer, self).__init__(mime_type='image/svg+xml', b64_encode=False, format='svg', width=width, height=height, scale=scale, engine=engine)