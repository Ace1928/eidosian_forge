import os
import warnings
import pkgutil
from plotly.optional_imports import get_module
from plotly import tools
from ._plotlyjs_version import __plotlyjs_version__
def _build_resize_script(plotdivid, plotly_root='Plotly'):
    resize_script = '<script type="text/javascript">window.addEventListener("resize", function(){{if (document.getElementById("{id}")) {{{plotly_root}.Plots.resize(document.getElementById("{id}"));}};}})</script>'.format(plotly_root=plotly_root, id=plotdivid)
    return resize_script