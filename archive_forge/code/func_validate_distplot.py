from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
def validate_distplot(hist_data, curve_type):
    """
    Distplot-specific validations

    :raises: (PlotlyError) If hist_data is not a list of lists
    :raises: (PlotlyError) If curve_type is not valid (i.e. not 'kde' or
        'normal').
    """
    hist_data_types = (list,)
    if np:
        hist_data_types += (np.ndarray,)
    if pd:
        hist_data_types += (pd.core.series.Series,)
    if not isinstance(hist_data[0], hist_data_types):
        raise exceptions.PlotlyError('Oops, this function was written to handle multiple datasets, if you want to plot just one, make sure your hist_data variable is still a list of lists, i.e. x = [1, 2, 3] -> x = [[1, 2, 3]]')
    curve_opts = ('kde', 'normal')
    if curve_type not in curve_opts:
        raise exceptions.PlotlyError("curve_type must be defined as 'kde' or 'normal'")
    if not scipy:
        raise ImportError('FigureFactory.create_distplot requires scipy')