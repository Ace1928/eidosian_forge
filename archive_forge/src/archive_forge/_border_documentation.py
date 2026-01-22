from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

        Construct a new Border object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.pointcloud.marker.Border`
        arearatio
            Specifies what fraction of the marker area is covered
            with the border.
        color
            Sets the stroke color. It accepts a specific color. If
            the color is not fully opaque and there are hundreds of
            thousands of points, it may cause slower zooming and
            panning.

        Returns
        -------
        Border
        