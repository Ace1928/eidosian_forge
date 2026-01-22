import hashlib
from pathlib import Path
from docutils import nodes
from docutils.parsers.rst import Directive, directives
import sphinx
from sphinx.errors import ConfigError, ExtensionError
import matplotlib as mpl
from matplotlib import _api, mathtext
from matplotlib.rcsetup import validate_float_or_None
class MathDirective(Directive):
    """
    The ``.. mathmpl::`` directive, as documented in the module's docstring.
    """
    has_content = True
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {'fontset': fontset_choice, 'fontsize': validate_float_or_None}

    def run(self):
        latex = ''.join(self.content)
        node = latex_math(self.block_text)
        node['latex'] = latex
        node['fontset'] = self.options.get('fontset', 'cm')
        node['fontsize'] = self.options.get('fontsize', setup.app.config.mathmpl_fontsize)
        return [node]