from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

        Construct a new Domain object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.layout.ternary.Domain`
        column
            If there is a layout grid, use the domain for this
            column in the grid for this ternary subplot .
        row
            If there is a layout grid, use the domain for this row
            in the grid for this ternary subplot .
        x
            Sets the horizontal domain of this ternary subplot (in
            plot fraction).
        y
            Sets the vertical domain of this ternary subplot (in
            plot fraction).

        Returns
        -------
        Domain
        