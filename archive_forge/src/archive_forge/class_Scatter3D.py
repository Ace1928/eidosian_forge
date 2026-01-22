import param
from ..core import Dimension, Element3D
from .geom import Points
from .path import Path
from .raster import Image
class Scatter3D(Element3D, Points):
    """
    Scatter3D is a 3D element representing the position of a collection
    of coordinates in a 3D space. The key dimensions represent the
    position of each coordinate along the x-, y- and z-axis.

    Scatter3D is not available for the default Bokeh backend.

    Example - Matplotlib
    --------------------

    .. code-block::

        import holoviews as hv
        from bokeh.sampledata.iris import flowers

        hv.extension("matplotlib")

        hv.Scatter3D(
            flowers, kdims=["sepal_length", "sepal_width", "petal_length"]
        ).opts(
            color="petal_width",
            alpha=0.7,
            size=5,
            cmap="fire",
            marker='^'
        )

    Example - Plotly
    ----------------

    .. code-block::

        import holoviews as hv
        from bokeh.sampledata.iris import flowers

        hv.extension("plotly")

        hv.Scatter3D(
            flowers, kdims=["sepal_length", "sepal_width", "petal_length"]
        ).opts(
            color="petal_width",
            alpha=0.7,
            size=5,
            cmap="Portland",
            colorbar=True,
            marker="circle",
        )
    """
    kdims = param.List(default=[Dimension('x'), Dimension('y'), Dimension('z')], bounds=(3, 3))
    vdims = param.List(default=[], doc='\n        Scatter3D can have optional value dimensions,\n        which may be mapped onto color and size.')
    group = param.String(default='Scatter3D', constant=True)

    def __getitem__(self, slc):
        return Points.__getitem__(self, slc)