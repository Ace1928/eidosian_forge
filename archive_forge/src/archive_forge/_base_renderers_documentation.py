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

    Renderer to display interactive figures in an external web browser.
    This renderer will open a new browser window or tab when the
    plotly.io.show function is called on a figure.

    This renderer has no ipython/jupyter dependencies and is a good choice
    for use in environments that do not support the inline display of
    interactive figures.

    mime type: 'text/html'
    