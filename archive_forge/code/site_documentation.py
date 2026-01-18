import sys
import os
import builtins
import _sitebuiltins
import io
import stat
Add standard site-specific directories to the module search path.

    This function is called automatically when this module is imported,
    unless the python interpreter was started with the -S flag.
    