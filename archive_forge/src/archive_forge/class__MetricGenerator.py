from functools import partial
import numpy as np
from . import _catboost
class _MetricGenerator(type):

    def __new__(mcs, name, parents, attrs):
        for k in attrs['_valid_params']:
            attrs[k] = property(partial(_get_param, name=k), partial(_set_param, name=k), partial(_del_param, name=k), 'Parameter {} of metric {}'.format(k, name))
        attrs['params_with_defaults'] = staticmethod(lambda: {param: {'default_value': default_value, 'is_mandatory': attrs['_is_mandatory_param'][param]} for param, default_value in attrs['_valid_params'].items()})
        docstring = ["Builtin metric: '{}'".format(name), 'Parameters:']
        if not attrs['_valid_params']:
            docstring[-1] += ' none'
        for param, value in attrs['_valid_params'].items():
            if not attrs['_is_mandatory_param'][param]:
                docstring.append(' ' * 4 + '{} = {} (default value)'.format(param, repr(value)))
            else:
                docstring.append(' ' * 4 + '{} (mandatory)'.format(param))
        attrs['__doc__'] = '\n'.join(docstring)
        attrs['__repr__'] = lambda self: '{}({})'.format(self._underlying_metric_name, ', '.join(['{}={} [mandatory={}]'.format(param, repr(value), self._is_mandatory_param[param]) for param, value in _current_params(self, False).items()]))
        attrs['__str__'] = _to_string

        def set_hints(self, **hints):
            for hint_key, hint_value in hints.items():
                if isinstance(hint_value, bool):
                    hints[hint_key] = str(hint_value).lower()
            setattr(self, 'hints', '|'.join(['{}~{}'.format(hint_key, hint_value) for hint_key, hint_value in hints.items()]))
            if 'hints' not in self._params:
                self._params.append('hints')
            return self
        attrs['set_hints'] = set_hints
        cls = super(_MetricGenerator, mcs).__new__(mcs, name, parents, attrs)
        return cls

    def __call__(cls, **kwargs):
        metric_obj = cls.__new__(cls)
        params = {k: v for k, v in cls._valid_params.items()}
        param_is_set = {param: not mandatory for param, mandatory in cls._is_mandatory_param.items()}
        for param, value in kwargs.items():
            if param not in cls._valid_params:
                raise ValueError('Unexpected parameter {}'.format(param))
            params[param] = value
            param_is_set[param] = True
        for param, is_set in param_is_set.items():
            if not is_set:
                raise ValueError('Parameter {} is mandatory and must be specified.'.format(param))
        for param, value in params.items():
            _set_param(metric_obj, value, param)
        metric_obj._params = list(params.keys())
        return metric_obj

    def __repr__(cls):
        return cls.__doc__

    def __setattr__(cls, name, value):
        if name in ('_valid_params', '_is_mandatory_param'):
            raise ValueError("Metric's `{}` shouldn't be modified or deleted.".format(name))
        type.__setattr__(cls, name, value)

    def __delattr__(cls, name):
        if name in ('_valid_params', '_is_mandatory_param'):
            raise ValueError("Metric's `{}` shouldn't be modified or deleted.".format(name))
        type.__delattr__(cls, name)