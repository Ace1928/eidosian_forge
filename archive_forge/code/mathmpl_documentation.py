import hashlib
from pathlib import Path
from docutils import nodes
from docutils.parsers.rst import Directive, directives
import sphinx
from sphinx.errors import ConfigError, ExtensionError
import matplotlib as mpl
from matplotlib import _api, mathtext
from matplotlib.rcsetup import validate_float_or_None

    The ``.. mathmpl::`` directive, as documented in the module's docstring.
    