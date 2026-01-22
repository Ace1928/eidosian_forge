from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class GraticuleGenerator(Generator):
    """GraticuleGenerator schema wrapper

    Parameters
    ----------

    graticule : bool, dict, :class:`GraticuleParams`
        Generate graticule GeoJSON data for geographic reference lines.
    name : str
        Provide a placeholder name and bind data at runtime.
    """
    _schema = {'$ref': '#/definitions/GraticuleGenerator'}

    def __init__(self, graticule: Union[bool, dict, 'SchemaBase', UndefinedType]=Undefined, name: Union[str, UndefinedType]=Undefined, **kwds):
        super(GraticuleGenerator, self).__init__(graticule=graticule, name=name, **kwds)