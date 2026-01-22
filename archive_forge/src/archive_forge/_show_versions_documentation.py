import platform
import sys
from .. import __version__
from ..utils.fixes import threadpool_info
from ._openmp_helpers import _openmp_parallelism_enabled
Print useful debugging information"

    .. versionadded:: 0.20

    Examples
    --------
    >>> from sklearn import show_versions
    >>> show_versions()  # doctest: +SKIP
    