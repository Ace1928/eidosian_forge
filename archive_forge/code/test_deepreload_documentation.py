import types
from pathlib import Path
import pytest
from tempfile import TemporaryDirectory
from IPython.lib.deepreload import modules_reloading
from IPython.lib.deepreload import reload as dreload
from IPython.utils.syspathcontext import prepended_to_syspath
Test that dreload does deep reloads and skips excluded modules.