import warnings
from math import sqrt
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs
from ..base import MultiOutputMixin, RegressorMixin, _fit_context
from ..model_selection import check_cv
from ..utils import Bunch, as_float_array, check_array
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.metadata_routing import (
from ..utils.parallel import Parallel, delayed
from ._base import LinearModel, _pre_fit
Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        