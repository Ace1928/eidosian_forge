from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

        Construct a new Lonaxis object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.layout.geo.Lonaxis`
        dtick
            Sets the graticule's longitude/latitude tick step.
        gridcolor
            Sets the graticule's stroke color.
        griddash
            Sets the dash style of lines. Set to a dash type string
            ("solid", "dot", "dash", "longdash", "dashdot", or
            "longdashdot") or a dash length list in px (eg
            "5px,10px,2px,2px").
        gridwidth
            Sets the graticule's stroke width (in px).
        range
            Sets the range of this axis (in degrees), sets the
            map's clipped coordinates.
        showgrid
            Sets whether or not graticule are shown on the map.
        tick0
            Sets the graticule's starting tick longitude/latitude.

        Returns
        -------
        Lonaxis
        