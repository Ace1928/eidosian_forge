from collections.abc import Sequence
from plotly import exceptions
from plotly.colors import (
def validate_dataframe(array):
    """
    Validates all strings or numbers in each dataframe column

    :raises: (PlotlyError) If there are any two items in any list whose
        types differ
    """
    from numbers import Number
    for vector in array:
        if isinstance(vector[0], Number):
            if not all((isinstance(item, Number) for item in vector)):
                raise exceptions.PlotlyError('Error in dataframe. Make sure all entries of each column are either numbers or strings.')
        elif isinstance(vector[0], str):
            if not all((isinstance(item, str) for item in vector)):
                raise exceptions.PlotlyError('Error in dataframe. Make sure all entries of each column are either numbers or strings.')