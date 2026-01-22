from rpy2 import robjects
from rpy2.robjects.lib import ggplot2, grdevices
from IPython import get_ipython  # type: ignore
from IPython.core.display import Image  # type: ignore
class GGPlotSVG(ggplot2.GGPlot):
    """ The embedding of several SVG figures into one ipython notebook is
    giving garbled figures. The SVG functionality is taken out to a
    child class.
    """

    def svg(self, width=6, height=4):
        """ Build an Ipython "Image" (requires iPython). """
        with grdevices.render_to_bytesio(grdevices.svg, width=width, height=height) as b:
            robjects.r('print')(self)
            data = b.getvalue()
            ip_img = Image(data=data, format='svg', embed=False)
            return ip_img