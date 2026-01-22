import os
from traitlets import (
from jupyter_core.paths import jupyter_path
from jupyter_server.transutils import _i18n
from jupyter_server.utils import url_path_join
@default('mathjax_url')
def _default_mathjax_url(self):
    if not self.enable_mathjax:
        return u''
    static_url_prefix = self.static_url_prefix
    return url_path_join(static_url_prefix, 'components', 'MathJax', 'MathJax.js')