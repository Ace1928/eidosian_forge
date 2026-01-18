from plotly.utils import _list_repr_elided
@property
def ys(self):
    """
        list of y-axis coordinates of each point in the lasso selection
        boundary

        Returns
        -------
        list[float]
        """
    return self._ys