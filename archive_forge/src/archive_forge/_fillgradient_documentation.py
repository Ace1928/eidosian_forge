from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

        Construct a new Fillgradient object

        Sets a fill gradient. If not specified, the fillcolor is used
        instead.

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.scatter.Fillgradient`
        colorscale
            Sets the fill gradient colors as a color scale. The
            color scale is interpreted as a gradient applied in the
            direction specified by "orientation", from the lowest
            to the highest value of the scatter plot along that
            axis, or from the center to the most distant point from
            it, if orientation is "radial".
        start
            Sets the gradient start value. It is given as the
            absolute position on the axis determined by the
            orientiation. E.g., if orientation is "horizontal", the
            gradient will be horizontal and start from the
            x-position given by start. If omitted, the gradient
            starts at the lowest value of the trace along the
            respective axis. Ignored if orientation is "radial".
        stop
            Sets the gradient end value. It is given as the
            absolute position on the axis determined by the
            orientiation. E.g., if orientation is "horizontal", the
            gradient will be horizontal and end at the x-position
            given by end. If omitted, the gradient ends at the
            highest value of the trace along the respective axis.
            Ignored if orientation is "radial".
        type
            Sets the type/orientation of the color gradient for the
            fill. Defaults to "none".

        Returns
        -------
        Fillgradient
        