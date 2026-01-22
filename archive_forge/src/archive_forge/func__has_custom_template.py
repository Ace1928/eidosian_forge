import argparse
import locale
import os
import sys
import time
from collections import OrderedDict
from os import path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from docutils.utils import column_width
import sphinx.locale
from sphinx import __display_version__, package_dir
from sphinx.locale import __
from sphinx.util.console import bold, color_terminal, colorize, nocolor, red  # type: ignore
from sphinx.util.osutil import ensuredir
from sphinx.util.template import SphinxRenderer
def _has_custom_template(self, template_name: str) -> bool:
    """Check if custom template file exists.

        Note: Please don't use this function from extensions.
              It will be removed in the future without deprecation period.
        """
    template = path.join(self.templatedir, path.basename(template_name))
    if self.templatedir and path.exists(template):
        return True
    else:
        return False