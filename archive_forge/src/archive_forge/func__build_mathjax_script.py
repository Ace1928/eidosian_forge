import os
import warnings
import pkgutil
from plotly.optional_imports import get_module
from plotly import tools
from ._plotlyjs_version import __plotlyjs_version__
def _build_mathjax_script(url):
    return '<script src="{url}?config=TeX-AMS-MML_SVG"></script>'.format(url=url)