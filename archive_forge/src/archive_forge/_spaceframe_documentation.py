from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

        Construct a new Spaceframe object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.volume.Spaceframe`
        fill
            Sets the fill ratio of the `spaceframe` elements. The
            default fill value is 1 meaning that they are entirely
            shaded. Applying a `fill` ratio less than one would
            allow the creation of openings parallel to the edges.
        show
            Displays/hides tetrahedron shapes between minimum and
            maximum iso-values. Often useful when either caps or
            surfaces are disabled or filled with values less than
            1.

        Returns
        -------
        Spaceframe
        