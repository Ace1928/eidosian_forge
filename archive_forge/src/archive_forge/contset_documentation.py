import logging
import bisect
from pyomo.common.numeric_types import native_numeric_types
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.set import SortedScalarSet
from pyomo.core.base.component import ModelComponentFactory
Returns the index of the nearest point in the
        :py:class:`ContinuousSet <pyomo.dae.ContinuousSet>`.

        If a tolerance is specified, the index will only be returned
        if the distance between the target and the closest point is
        less than or equal to that tolerance. If there is a tie for
        closest point, the index on the left is returned.

        Parameters
        ----------
        target : `float`
        tolerance : `float` or `None`

        Returns
        -------
        `float` or `None`
        