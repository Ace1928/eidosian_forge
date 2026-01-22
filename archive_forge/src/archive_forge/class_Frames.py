import warnings
class Frames(list):
    """
        plotly.graph_objs.Frames is deprecated.
    Please replace it with a list or tuple of instances of the following types
      - plotly.graph_objs.Frame

    """

    def __init__(self, *args, **kwargs):
        """
                plotly.graph_objs.Frames is deprecated.
        Please replace it with a list or tuple of instances of the following types
          - plotly.graph_objs.Frame

        """
        warnings.warn('plotly.graph_objs.Frames is deprecated.\nPlease replace it with a list or tuple of instances of the following types\n  - plotly.graph_objs.Frame\n', DeprecationWarning)
        super(Frames, self).__init__(*args, **kwargs)