from warnings import warn
import numpy as np
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.evaluation import after_stat
from .binning import (
from .stat import stat

    Count cases in each interval

    {usage}

    Parameters
    ----------
    {common_parameters}
    binwidth : float, default=None
        The width of the bins. The default is to use bins bins that
        cover the range of the data. You should always override this
        value, exploring multiple widths to find the best to illustrate
        the stories in your data.
    bins : int, default=None
        Number of bins. Overridden by binwidth. If `None`{.py},
        a number is computed using the freedman-diaconis method.
    breaks : array_like, default=None
        Bin boundaries. This supercedes the `binwidth`, `bins`,
        `center` and `boundary`.
    center : float, default=None
        The center of one of the bins. Note that if center is above
        or below the range of the data, things will be shifted by
        an appropriate number of widths. To center on integers, for
        example, use `width=1`{.py} and `center=0`{.py}, even if 0 i
        s outside the range of the data. At most one of center and
        boundary may be specified.
    boundary : float, default=None
        A boundary between two bins. As with center, things are
        shifted when boundary is outside the range of the data.
        For example, to center on integers, use `width=1`{.py} and
        `boundary=0.5`{.py}, even if 1 is outside the range of the
        data. At most one of center and boundary may be specified.
    closed : Literal["left", "right"], default="right"
        Which edge of the bins is included.
    pad : bool, default=False
        If `True`{.py}, adds empty bins at either side of x.
        This ensures that frequency polygons touch 0.
    