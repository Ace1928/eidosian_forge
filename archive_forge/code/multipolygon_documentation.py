import shapely
from shapely.geometry import polygon
from shapely.geometry.base import BaseMultipartGeometry
Returns group of SVG path elements for the MultiPolygon geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is to use "#66cc99" if
            geometry is valid, and "#ff3333" if invalid.
        opacity : float
            Float number between 0 and 1 for color opacity. Default value is 0.6
        