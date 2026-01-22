from io import BytesIO, open
import os
import tempfile
import shutil
import subprocess
from base64 import encodebytes
import textwrap
from pathlib import Path
from IPython.utils.process import find_cmd, FindCmdError
from traitlets.config import get_config
from traitlets.config.configurable import SingletonConfigurable
from traitlets import List, Bool, Unicode
from IPython.utils.py3compat import cast_unicode
class LaTeXTool(SingletonConfigurable):
    """An object to store configuration of the LaTeX tool."""

    def _config_default(self):
        return get_config()
    backends = List(Unicode(), ['matplotlib', 'dvipng'], help='Preferred backend to draw LaTeX math equations. Backends in the list are checked one by one and the first usable one is used.  Note that `matplotlib` backend is usable only for inline style equations.  To draw  display style equations, `dvipng` backend must be specified. ').tag(config=True)
    use_breqn = Bool(True, help='Use breqn.sty to automatically break long equations. This configuration takes effect only for dvipng backend.').tag(config=True)
    packages = List(['amsmath', 'amsthm', 'amssymb', 'bm'], help="A list of packages to use for dvipng backend. 'breqn' will be automatically appended when use_breqn=True.").tag(config=True)
    preamble = Unicode(help='Additional preamble to use when generating LaTeX source for dvipng backend.').tag(config=True)