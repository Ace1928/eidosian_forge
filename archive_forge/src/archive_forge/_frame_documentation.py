from plotly.basedatatypes import BaseFrameHierarchyType as _BaseFrameHierarchyType
import copy as _copy

        Construct a new Frame object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of :class:`plotly.graph_objs.Frame`
        baseframe
            The name of the frame into which this frame's
            properties are merged before applying. This is used to
            unify properties and avoid needing to specify the same
            values for the same properties in multiple frames.
        data
            A list of traces this frame modifies. The format is
            identical to the normal trace definition.
        group
            An identifier that specifies the group to which the
            frame belongs, used by animate to select a subset of
            frames.
        layout
            Layout properties which this frame modifies. The format
            is identical to the normal layout definition.
        name
            A label by which to identify the frame
        traces
            A list of trace indices that identify the respective
            traces in the data attribute

        Returns
        -------
        Frame
        