from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy
class Tickformatstop(_BaseTraceHierarchyType):
    _parent_path_str = 'surface.colorbar'
    _path_str = 'surface.colorbar.tickformatstop'
    _valid_props = {'dtickrange', 'enabled', 'name', 'templateitemname', 'value'}

    @property
    def dtickrange(self):
        """
            range [*min*, *max*], where "min", "max" - dtick values which
            describe some zoom level, it is possible to omit "min" or "max"
            value by passing "null"

            The 'dtickrange' property is an info array that may be specified as:

            * a list or tuple of 2 elements where:
        (0) The 'dtickrange[0]' property accepts values of any type
        (1) The 'dtickrange[1]' property accepts values of any type

            Returns
            -------
            list
        """
        return self['dtickrange']

    @dtickrange.setter
    def dtickrange(self, val):
        self['dtickrange'] = val

    @property
    def enabled(self):
        """
        Determines whether or not this stop is used. If `false`, this
        stop is ignored even within its `dtickrange`.

        The 'enabled' property must be specified as a bool
        (either True, or False)

        Returns
        -------
        bool
        """
        return self['enabled']

    @enabled.setter
    def enabled(self, val):
        self['enabled'] = val

    @property
    def name(self):
        """
        When used in a template, named items are created in the output
        figure in addition to any items the figure already has in this
        array. You can modify these items in the output figure by
        making your own item with `templateitemname` matching this
        `name` alongside your modifications (including `visible: false`
        or `enabled: false` to hide it). Has no effect outside of a
        template.

        The 'name' property is a string and must be specified as:
          - A string
          - A number that will be converted to a string

        Returns
        -------
        str
        """
        return self['name']

    @name.setter
    def name(self, val):
        self['name'] = val

    @property
    def templateitemname(self):
        """
        Used to refer to a named item in this array in the template.
        Named items from the template will be created even without a
        matching item in the input figure, but you can modify one by
        making an item with `templateitemname` matching its `name`,
        alongside your modifications (including `visible: false` or
        `enabled: false` to hide it). If there is no template or no
        matching item, this item will be hidden unless you explicitly
        show it with `visible: true`.

        The 'templateitemname' property is a string and must be specified as:
          - A string
          - A number that will be converted to a string

        Returns
        -------
        str
        """
        return self['templateitemname']

    @templateitemname.setter
    def templateitemname(self, val):
        self['templateitemname'] = val

    @property
    def value(self):
        """
        string - dtickformat for described zoom level, the same as
        "tickformat"

        The 'value' property is a string and must be specified as:
          - A string
          - A number that will be converted to a string

        Returns
        -------
        str
        """
        return self['value']

    @value.setter
    def value(self, val):
        self['value'] = val

    @property
    def _prop_descriptions(self):
        return '        dtickrange\n            range [*min*, *max*], where "min", "max" - dtick values\n            which describe some zoom level, it is possible to omit\n            "min" or "max" value by passing "null"\n        enabled\n            Determines whether or not this stop is used. If\n            `false`, this stop is ignored even within its\n            `dtickrange`.\n        name\n            When used in a template, named items are created in the\n            output figure in addition to any items the figure\n            already has in this array. You can modify these items\n            in the output figure by making your own item with\n            `templateitemname` matching this `name` alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). Has no effect outside of a\n            template.\n        templateitemname\n            Used to refer to a named item in this array in the\n            template. Named items from the template will be created\n            even without a matching item in the input figure, but\n            you can modify one by making an item with\n            `templateitemname` matching its `name`, alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). If there is no template or no\n            matching item, this item will be hidden unless you\n            explicitly show it with `visible: true`.\n        value\n            string - dtickformat for described zoom level, the same\n            as "tickformat"\n        '

    def __init__(self, arg=None, dtickrange=None, enabled=None, name=None, templateitemname=None, value=None, **kwargs):
        """
        Construct a new Tickformatstop object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of :class:`plotly.graph_objs.surface.colorb
            ar.Tickformatstop`
        dtickrange
            range [*min*, *max*], where "min", "max" - dtick values
            which describe some zoom level, it is possible to omit
            "min" or "max" value by passing "null"
        enabled
            Determines whether or not this stop is used. If
            `false`, this stop is ignored even within its
            `dtickrange`.
        name
            When used in a template, named items are created in the
            output figure in addition to any items the figure
            already has in this array. You can modify these items
            in the output figure by making your own item with
            `templateitemname` matching this `name` alongside your
            modifications (including `visible: false` or `enabled:
            false` to hide it). Has no effect outside of a
            template.
        templateitemname
            Used to refer to a named item in this array in the
            template. Named items from the template will be created
            even without a matching item in the input figure, but
            you can modify one by making an item with
            `templateitemname` matching its `name`, alongside your
            modifications (including `visible: false` or `enabled:
            false` to hide it). If there is no template or no
            matching item, this item will be hidden unless you
            explicitly show it with `visible: true`.
        value
            string - dtickformat for described zoom level, the same
            as "tickformat"

        Returns
        -------
        Tickformatstop
        """
        super(Tickformatstop, self).__init__('tickformatstops')
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
            raise ValueError('The first argument to the plotly.graph_objs.surface.colorbar.Tickformatstop\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.surface.colorbar.Tickformatstop`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('dtickrange', None)
        _v = dtickrange if dtickrange is not None else _v
        if _v is not None:
            self['dtickrange'] = _v
        _v = arg.pop('enabled', None)
        _v = enabled if enabled is not None else _v
        if _v is not None:
            self['enabled'] = _v
        _v = arg.pop('name', None)
        _v = name if name is not None else _v
        if _v is not None:
            self['name'] = _v
        _v = arg.pop('templateitemname', None)
        _v = templateitemname if templateitemname is not None else _v
        if _v is not None:
            self['templateitemname'] = _v
        _v = arg.pop('value', None)
        _v = value if value is not None else _v
        if _v is not None:
            self['value'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False