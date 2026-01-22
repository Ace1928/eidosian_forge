from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

        Construct a new Symbol object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.layout.mapbox.layer.Symbol`
        icon
            Sets the symbol icon image (mapbox.layer.layout.icon-
            image). Full list: https://www.mapbox.com/maki-icons/
        iconsize
            Sets the symbol icon size (mapbox.layer.layout.icon-
            size). Has an effect only when `type` is set to
            "symbol".
        placement
            Sets the symbol and/or text placement
            (mapbox.layer.layout.symbol-placement). If `placement`
            is "point", the label is placed where the geometry is
            located If `placement` is "line", the label is placed
            along the line of the geometry If `placement` is "line-
            center", the label is placed on the center of the
            geometry
        text
            Sets the symbol text (mapbox.layer.layout.text-field).
        textfont
            Sets the icon text font (color=mapbox.layer.paint.text-
            color, size=mapbox.layer.layout.text-size). Has an
            effect only when `type` is set to "symbol".
        textposition
            Sets the positions of the `text` elements with respects
            to the (x,y) coordinates.

        Returns
        -------
        Symbol
        