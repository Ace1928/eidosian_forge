from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class NamedData(DataSource):
    """NamedData schema wrapper

    Parameters
    ----------

    name : str
        Provide a placeholder name and bind data at runtime.

        New data may change the layout but Vega does not always resize the chart. To update
        the layout when the data updates, set `autosize
        <https://vega.github.io/vega-lite/docs/size.html#autosize>`__ or explicitly use
        `view.resize <https://vega.github.io/vega/docs/api/view/#view_resize>`__.
    format : dict, :class:`DataFormat`, :class:`CsvDataFormat`, :class:`DsvDataFormat`, :class:`JsonDataFormat`, :class:`TopoDataFormat`
        An object that specifies the format for parsing the data.
    """
    _schema = {'$ref': '#/definitions/NamedData'}

    def __init__(self, name: Union[str, UndefinedType]=Undefined, format: Union[dict, 'SchemaBase', UndefinedType]=Undefined, **kwds):
        super(NamedData, self).__init__(name=name, format=format, **kwds)