from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

        Construct a new Ternary object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.layout.Ternary`
        aaxis
            :class:`plotly.graph_objects.layout.ternary.Aaxis`
            instance or dict with compatible properties
        baxis
            :class:`plotly.graph_objects.layout.ternary.Baxis`
            instance or dict with compatible properties
        bgcolor
            Set the background color of the subplot
        caxis
            :class:`plotly.graph_objects.layout.ternary.Caxis`
            instance or dict with compatible properties
        domain
            :class:`plotly.graph_objects.layout.ternary.Domain`
            instance or dict with compatible properties
        sum
            The number each triplet should sum to, and the maximum
            range of each axis
        uirevision
            Controls persistence of user-driven changes in axis
            `min` and `title`, if not overridden in the individual
            axes. Defaults to `layout.uirevision`.

        Returns
        -------
        Ternary
        