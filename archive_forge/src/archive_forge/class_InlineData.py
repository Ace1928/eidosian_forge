from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class InlineData(DataSource):
    """InlineData schema wrapper

    Parameters
    ----------

    values : str, dict, Sequence[str], Sequence[bool], Sequence[dict], Sequence[float], :class:`InlineDataset`
        The full data set, included inline. This can be an array of objects or primitive
        values, an object, or a string. Arrays of primitive values are ingested as objects
        with a ``data`` property. Strings are parsed according to the specified format type.
    format : dict, :class:`DataFormat`, :class:`CsvDataFormat`, :class:`DsvDataFormat`, :class:`JsonDataFormat`, :class:`TopoDataFormat`
        An object that specifies the format for parsing the data.
    name : str
        Provide a placeholder name and bind data at runtime.
    """
    _schema = {'$ref': '#/definitions/InlineData'}

    def __init__(self, values: Union[str, dict, 'SchemaBase', Sequence[str], Sequence[bool], Sequence[dict], Sequence[float], UndefinedType]=Undefined, format: Union[dict, 'SchemaBase', UndefinedType]=Undefined, name: Union[str, UndefinedType]=Undefined, **kwds):
        super(InlineData, self).__init__(values=values, format=format, name=name, **kwds)