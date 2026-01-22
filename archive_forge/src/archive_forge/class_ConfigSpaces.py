from collections import OrderedDict
import numpy as _np
class ConfigSpaces(object):
    """The configuration spaces of all ops."""

    def __init__(self):
        self.spaces = {}

    def __setitem__(self, name, space):
        self.spaces[name] = space

    def __len__(self):
        return len(self.spaces)

    def __repr__(self):
        res = 'ConfigSpaces (len=%d, config_space=\n' % len(self)
        for i, (key, val) in enumerate(self.spaces.items()):
            res += '  %2d %s:\n %s\n' % (i, key, val)
        return res + ')'

    def to_json_dict(self):
        """convert to a json serializable dictionary

        Return
        ------
        ret: dict
            a json serializable dictionary
        """
        ret = []
        for k, v in self.spaces.items():
            ret.append((k, v.to_json_dict()))
        return ret

    @classmethod
    def from_json_dict(cls, json_dict):
        """Build a ConfigSpaces from json serializable dictionary

        Parameters
        ----------
        cls: class
            The calling class
        json_dict: dict
            Json serializable dictionary.

        Returns
        -------
        ret: ConfigSpaces
            The corresponding ConfigSpaces object
        """
        ret = cls()
        for key, val in json_dict:
            ret.spaces[key] = ConfigSpace.from_json_dict(val)
        return ret