import sys
import re
import os
from configparser import RawConfigParser
class VariableSet:
    """
    Container object for the variables defined in a config file.

    `VariableSet` can be used as a plain dictionary, with the variable names
    as keys.

    Parameters
    ----------
    d : dict
        Dict of items in the "variables" section of the configuration file.

    """

    def __init__(self, d):
        self._raw_data = dict([(k, v) for k, v in d.items()])
        self._re = {}
        self._re_sub = {}
        self._init_parse()

    def _init_parse(self):
        for k, v in self._raw_data.items():
            self._init_parse_var(k, v)

    def _init_parse_var(self, name, value):
        self._re[name] = re.compile('\\$\\{%s\\}' % name)
        self._re_sub[name] = value

    def interpolate(self, value):

        def _interpolate(value):
            for k in self._re.keys():
                value = self._re[k].sub(self._re_sub[k], value)
            return value
        while _VAR.search(value):
            nvalue = _interpolate(value)
            if nvalue == value:
                break
            value = nvalue
        return value

    def variables(self):
        """
        Return the list of variable names.

        Parameters
        ----------
        None

        Returns
        -------
        names : list of str
            The names of all variables in the `VariableSet` instance.

        """
        return list(self._raw_data.keys())

    def __getitem__(self, name):
        return self._raw_data[name]

    def __setitem__(self, name, value):
        self._raw_data[name] = value
        self._init_parse_var(name, value)