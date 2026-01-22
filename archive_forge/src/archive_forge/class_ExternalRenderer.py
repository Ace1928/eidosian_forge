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
class ExternalRenderer(BaseRenderer):
    """
    Base class for external renderers.  ExternalRenderer subclasses
    do not display figures inline in a notebook environment, but render
    figures by some external means (e.g. a separate browser tab).

    Unlike MimetypeRenderer subclasses, ExternalRenderer subclasses are not
    invoked when a figure is asked to display itself in the notebook.
    Instead, they are invoked when the plotly.io.show function is called
    on a figure.
    """

    def render(self, fig):
        raise NotImplementedError()