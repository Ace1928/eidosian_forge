import platform
import sys
from .. import __version__
from ..utils.fixes import threadpool_info
from ._openmp_helpers import _openmp_parallelism_enabled
def _get_sys_info():
    """System information

    Returns
    -------
    sys_info : dict
        system and Python version information

    """
    python = sys.version.replace('\n', ' ')
    blob = [('python', python), ('executable', sys.executable), ('machine', platform.platform())]
    return dict(blob)