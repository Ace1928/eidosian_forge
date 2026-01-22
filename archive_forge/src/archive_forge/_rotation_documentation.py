from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

        Construct a new Rotation object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of :class:`plotly.graph_objs.layout.geo.pro
            jection.Rotation`
        lat
            Rotates the map along meridians (in degrees North).
        lon
            Rotates the map along parallels (in degrees East).
            Defaults to the center of the `lonaxis.range` values.
        roll
            Roll the map (in degrees) For example, a roll of 180
            makes the map appear upside down.

        Returns
        -------
        Rotation
        