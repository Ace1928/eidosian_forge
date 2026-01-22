import os
from traitlets import (
from jupyter_core.paths import jupyter_path
from jupyter_server.transutils import _i18n
from jupyter_server.utils import url_path_join
@observe('enable_mathjax')
def _update_enable_mathjax(self, change):
    """set mathjax url to empty if mathjax is disabled"""
    if not change['new']:
        self.mathjax_url = u''