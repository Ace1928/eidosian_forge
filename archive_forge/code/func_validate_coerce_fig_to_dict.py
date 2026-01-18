import plotly
import plotly.graph_objs as go
from plotly.offline import get_plotlyjs_version
def validate_coerce_fig_to_dict(fig, validate):
    from plotly.basedatatypes import BaseFigure
    if isinstance(fig, BaseFigure):
        fig_dict = fig.to_dict()
    elif isinstance(fig, dict):
        if validate:
            fig_dict = plotly.graph_objs.Figure(fig).to_plotly_json()
        else:
            fig_dict = fig
    elif hasattr(fig, 'to_plotly_json'):
        fig_dict = fig.to_plotly_json()
    else:
        raise ValueError('\nThe fig parameter must be a dict or Figure.\n    Received value of type {typ}: {v}'.format(typ=type(fig), v=fig))
    return fig_dict