import scipy as sp
from ...utils._plotting import _BinaryClassifierCurveDisplayMixin
from .._ranking import det_curve
Plot visualization.

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        name : str, default=None
            Name of DET curve for labeling. If `None`, use `estimator_name` if
            it is not `None`, otherwise no labeling is shown.

        **kwargs : dict
            Additional keywords arguments passed to matplotlib `plot` function.

        Returns
        -------
        display : :class:`~sklearn.metrics.DetCurveDisplay`
            Object that stores computed values.
        