import atexit
import shutil
import tempfile
from ._version import __version__  # noqa
from .compilation import compile_stan_file, format_stan_file
from .install_cmdstan import rebuild_cmdstan
from .model import CmdStanModel
from .stanfit import (
from .utils import (
def _cleanup_tmpdir() -> None:
    """Force deletion of _TMPDIR."""
    shutil.rmtree(_TMPDIR, ignore_errors=True)