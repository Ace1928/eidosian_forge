from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

        Construct a new Gauge object

        The gauge of the Indicator plot.

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.indicator.Gauge`
        axis
            :class:`plotly.graph_objects.indicator.gauge.Axis`
            instance or dict with compatible properties
        bar
            Set the appearance of the gauge's value
        bgcolor
            Sets the gauge background color.
        bordercolor
            Sets the color of the border enclosing the gauge.
        borderwidth
            Sets the width (in px) of the border enclosing the
            gauge.
        shape
            Set the shape of the gauge
        steps
            A tuple of
            :class:`plotly.graph_objects.indicator.gauge.Step`
            instances or dicts with compatible properties
        stepdefaults
            When used in a template (as
            layout.template.data.indicator.gauge.stepdefaults),
            sets the default property values to use for elements of
            indicator.gauge.steps
        threshold
            :class:`plotly.graph_objects.indicator.gauge.Threshold`
            instance or dict with compatible properties

        Returns
        -------
        Gauge
        