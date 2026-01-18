from plotly.utils import _list_repr_elided
@property
def trace_index(self):
    """
        Index of the trace in the figure

        Returns
        -------
        int
        """
    return self._trace_index