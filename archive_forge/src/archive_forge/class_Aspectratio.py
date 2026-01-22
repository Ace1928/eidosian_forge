from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy
class Aspectratio(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.scene'
    _path_str = 'layout.scene.aspectratio'
    _valid_props = {'x', 'y', 'z'}

    @property
    def x(self):
        """
        The 'x' property is a number and may be specified as:
          - An int or float in the interval [0, inf]

        Returns
        -------
        int|float
        """
        return self['x']

    @x.setter
    def x(self, val):
        self['x'] = val

    @property
    def y(self):
        """
        The 'y' property is a number and may be specified as:
          - An int or float in the interval [0, inf]

        Returns
        -------
        int|float
        """
        return self['y']

    @y.setter
    def y(self, val):
        self['y'] = val

    @property
    def z(self):
        """
        The 'z' property is a number and may be specified as:
          - An int or float in the interval [0, inf]

        Returns
        -------
        int|float
        """
        return self['z']

    @z.setter
    def z(self, val):
        self['z'] = val

    @property
    def _prop_descriptions(self):
        return '        x\n\n        y\n\n        z\n\n        '

    def __init__(self, arg=None, x=None, y=None, z=None, **kwargs):
        """
        Construct a new Aspectratio object

        Sets this scene's axis aspectratio.

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.layout.scene.Aspectratio`
        x

        y

        z


        Returns
        -------
        Aspectratio
        """
        super(Aspectratio, self).__init__('aspectratio')
        if '_parent' in kwargs:
            self._parent = kwargs['_parent']
            return
        if arg is None:
            arg = {}
        elif isinstance(arg, self.__class__):
            arg = arg.to_plotly_json()
        elif isinstance(arg, dict):
            arg = _copy.copy(arg)
        else:
            raise ValueError('The first argument to the plotly.graph_objs.layout.scene.Aspectratio\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.scene.Aspectratio`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('x', None)
        _v = x if x is not None else _v
        if _v is not None:
            self['x'] = _v
        _v = arg.pop('y', None)
        _v = y if y is not None else _v
        if _v is not None:
            self['y'] = _v
        _v = arg.pop('z', None)
        _v = z if z is not None else _v
        if _v is not None:
            self['z'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False