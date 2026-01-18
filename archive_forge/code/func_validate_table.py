from plotly import exceptions, optional_imports
from plotly.graph_objs import graph_objs
def validate_table(table_text, font_colors):
    """
    Table-specific validations

    Check that font_colors is supplied correctly (1, 3, or len(text)
        colors).

    :raises: (PlotlyError) If font_colors is supplied incorretly.

    See FigureFactory.create_table() for params
    """
    font_colors_len_options = [1, 3, len(table_text)]
    if len(font_colors) not in font_colors_len_options:
        raise exceptions.PlotlyError('Oops, font_colors should be a list of length 1, 3 or len(text)')