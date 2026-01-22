from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class UrlData(DataSource):
    """UrlData schema wrapper

    Parameters
    ----------

    url : str
        An URL from which to load the data set. Use the ``format.type`` property to ensure
        the loaded data is correctly parsed.
    format : dict, :class:`DataFormat`, :class:`CsvDataFormat`, :class:`DsvDataFormat`, :class:`JsonDataFormat`, :class:`TopoDataFormat`
        An object that specifies the format for parsing the data.
    name : str
        Provide a placeholder name and bind data at runtime.
    """
    _schema = {'$ref': '#/definitions/UrlData'}

    def __init__(self, url: Union[str, UndefinedType]=Undefined, format: Union[dict, 'SchemaBase', UndefinedType]=Undefined, name: Union[str, UndefinedType]=Undefined, **kwds):
        super(UrlData, self).__init__(url=url, format=format, name=name, **kwds)