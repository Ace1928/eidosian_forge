from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

        Construct a new Threshold object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.indicator.gauge.Threshold`
        line
            :class:`plotly.graph_objects.indicator.gauge.threshold.
            Line` instance or dict with compatible properties
        thickness
            Sets the thickness of the threshold line as a fraction
            of the thickness of the gauge.
        value
            Sets a treshold value drawn as a line.

        Returns
        -------
        Threshold
        