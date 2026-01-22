from collections import OrderedDict
import numpy as _np
class ConfigSpace(object):
    """The configuration space of a schedule."""

    def __init__(self, space_map, _entity_map):
        self.space_map = space_map
        self._entity_map = _entity_map
        self._length = None

    @classmethod
    def from_tvm(cls, x):
        """Build a ConfigSpace from autotvm.ConfigSpace

        Parameters
        ----------
        cls: class
            Calling class
        x: autotvm.ConfigSpace
            The source object

        Returns
        -------
        ret: ConfigSpace
            The corresponding ConfigSpace object
        """
        space_map = OrderedDict([(k, OtherOptionSpace.from_tvm(v)) for k, v in x.space_map.items()])
        _entity_map = OrderedDict([(k, OtherOptionEntity.from_tvm(v)) for k, v in x._entity_map.items()])
        return cls(space_map, _entity_map)

    def __len__(self):
        if self._length is None:
            self._length = int(_np.prod([len(x) for x in self.space_map.values()]))
        return self._length

    def __repr__(self):
        res = 'ConfigSpace (len=%d, space_map=\n' % len(self)
        for i, (name, space) in enumerate(self.space_map.items()):
            res += '  %2d %s: %s\n' % (i, name, space)
        return res + ')'

    def to_json_dict(self):
        """convert to a json serializable dictionary

        Return
        ------
        ret: dict
            a json serializable dictionary
        """
        ret = {}
        entity_map = []
        for k, v in self._entity_map.items():
            if isinstance(v, OtherOptionEntity):
                entity_map.append((k, 'ot', v.val))
            else:
                raise RuntimeError('Invalid entity instance: ' + v)
        ret['e'] = entity_map
        space_map = []
        for k, v in self.space_map.items():
            entities = [e.val for e in v.entities]
            space_map.append((k, 'ot', entities))
        ret['s'] = space_map
        return ret

    @classmethod
    def from_json_dict(cls, json_dict):
        """Build a ConfigSpace from json serializable dictionary

        Parameters
        ----------
        cls: class
            The calling class
        json_dict: dict
            Json serializable dictionary.

        Returns
        -------
        ret: ConfigSpace
            The corresponding ConfigSpace object
        """
        entity_map = OrderedDict()
        for item in json_dict['e']:
            key, knob_type, knob_args = item
            if knob_type == 'ot':
                entity = OtherOptionEntity(knob_args)
            else:
                raise RuntimeError('Invalid config knob type: ' + knob_type)
            entity_map[str(key)] = entity
        space_map = OrderedDict()
        for item in json_dict['s']:
            key, knob_type, knob_args = item
            if knob_type == 'ot':
                space = OtherOptionSpace(knob_args)
            else:
                raise RuntimeError('Invalid config knob type: ' + knob_type)
            space_map[str(key)] = space
        return cls(space_map, entity_map)