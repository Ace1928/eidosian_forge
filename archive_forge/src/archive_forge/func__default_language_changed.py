from within Jinja templates.
from html import escape
from warnings import warn
from traitlets import Dict, observe
from nbconvert.utils.base import NbConvertBase
@observe('default_language')
def _default_language_changed(self, change):
    warn('Setting default_language in config is deprecated as of 5.0, please use language_info metadata instead.', stacklevel=2)
    self.pygments_lexer = change['new']