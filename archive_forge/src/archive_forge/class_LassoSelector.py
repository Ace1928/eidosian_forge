from plotly.utils import _list_repr_elided
class LassoSelector:

    def __init__(self, xs=None, ys=None, **_):
        self._type = 'lasso'
        self._xs = xs
        self._ys = ys

    def __repr__(self):
        return 'LassoSelector(xs={xs},\n              ys={ys})'.format(xs=_list_repr_elided(self.xs, indent=len('LassoSelector(xs=')), ys=_list_repr_elided(self.ys, indent=len('              ys=')))

    @property
    def type(self):
        """
        The selector's type

        Returns
        -------
        str
        """
        return self._type

    @property
    def xs(self):
        """
        list of x-axis coordinates of each point in the lasso selection
        boundary

        Returns
        -------
        list[float]
        """
        return self._xs

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