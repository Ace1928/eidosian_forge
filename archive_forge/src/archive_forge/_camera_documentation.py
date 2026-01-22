from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

        Construct a new Camera object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.layout.scene.Camera`
        center
            Sets the (x,y,z) components of the 'center' camera
            vector This vector determines the translation (x,y,z)
            space about the center of this scene. By default, there
            is no such translation.
        eye
            Sets the (x,y,z) components of the 'eye' camera vector.
            This vector determines the view point about the origin
            of this scene.
        projection
            :class:`plotly.graph_objects.layout.scene.camera.Projec
            tion` instance or dict with compatible properties
        up
            Sets the (x,y,z) components of the 'up' camera vector.
            This vector determines the up direction of this scene
            with respect to the page. The default is *{x: 0, y: 0,
            z: 1}* which means that the z axis points up.

        Returns
        -------
        Camera
        