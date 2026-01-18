import plotly.colors as clrs
from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
from plotly.validators.heatmap import ColorscaleValidator
def validate_annotated_heatmap(z, x, y, annotation_text):
    """
    Annotated-heatmap-specific validations

    Check that if a text matrix is supplied, it has the same
    dimensions as the z matrix.

    See FigureFactory.create_annotated_heatmap() for params

    :raises: (PlotlyError) If z and text matrices do not  have the same
        dimensions.
    """
    if annotation_text is not None and isinstance(annotation_text, list):
        utils.validate_equal_length(z, annotation_text)
        for lst in range(len(z)):
            if len(z[lst]) != len(annotation_text[lst]):
                raise exceptions.PlotlyError('z and text should have the same dimensions')
    if x:
        if len(x) != len(z[0]):
            raise exceptions.PlotlyError('oops, the x list that you provided does not match the width of your z matrix ')
    if y:
        if len(y) != len(z):
            raise exceptions.PlotlyError('oops, the y list that you provided does not match the length of your z matrix ')