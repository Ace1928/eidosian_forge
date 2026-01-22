from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

        Construct a new Delta object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.indicator.Delta`
        decreasing
            :class:`plotly.graph_objects.indicator.delta.Decreasing
            ` instance or dict with compatible properties
        font
            Set the font used to display the delta
        increasing
            :class:`plotly.graph_objects.indicator.delta.Increasing
            ` instance or dict with compatible properties
        position
            Sets the position of delta with respect to the number.
        prefix
            Sets a prefix appearing before the delta.
        reference
            Sets the reference value to compute the delta. By
            default, it is set to the current value.
        relative
            Show relative change
        suffix
            Sets a suffix appearing next to the delta.
        valueformat
            Sets the value formatting rule using d3 formatting
            mini-languages which are very similar to those in
            Python. For numbers, see:
            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.

        Returns
        -------
        Delta
        