from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class ChainedAssignmentError(Warning):
    """
    Warning raised when trying to set using chained assignment.

    When the ``mode.copy_on_write`` option is enabled, chained assignment can
    never work. In such a situation, we are always setting into a temporary
    object that is the result of an indexing operation (getitem), which under
    Copy-on-Write always behaves as a copy. Thus, assigning through a chain
    can never update the original Series or DataFrame.

    For more information on view vs. copy,
    see :ref:`the user guide<indexing.view_versus_copy>`.

    Examples
    --------
    >>> pd.options.mode.copy_on_write = True
    >>> df = pd.DataFrame({'A': [1, 1, 1, 2, 2]}, columns=['A'])
    >>> df["A"][0:3] = 10 # doctest: +SKIP
    ... # ChainedAssignmentError: ...
    >>> pd.options.mode.copy_on_write = False
    """