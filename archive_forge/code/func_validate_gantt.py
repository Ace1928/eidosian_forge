from numbers import Number
import copy
from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
import plotly.graph_objects as go
def validate_gantt(df):
    """
    Validates the inputted dataframe or list
    """
    if pd and isinstance(df, pd.core.frame.DataFrame):
        for key in REQUIRED_GANTT_KEYS:
            if key not in df:
                raise exceptions.PlotlyError('The columns in your dataframe must include the following keys: {0}'.format(', '.join(REQUIRED_GANTT_KEYS)))
        num_of_rows = len(df.index)
        chart = []
        for index in range(num_of_rows):
            task_dict = {}
            for key in df:
                task_dict[key] = df.iloc[index][key]
            chart.append(task_dict)
        return chart
    if not isinstance(df, list):
        raise exceptions.PlotlyError('You must input either a dataframe or a list of dictionaries.')
    if len(df) <= 0:
        raise exceptions.PlotlyError('Your list is empty. It must contain at least one dictionary.')
    if not isinstance(df[0], dict):
        raise exceptions.PlotlyError('Your list must only include dictionaries.')
    return df