from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class SortByEncoding(Sort):
    """SortByEncoding schema wrapper

    Parameters
    ----------

    encoding : :class:`SortByChannel`, Literal['x', 'y', 'color', 'fill', 'stroke', 'strokeWidth', 'size', 'shape', 'fillOpacity', 'strokeOpacity', 'opacity', 'text']
        The `encoding channel
        <https://vega.github.io/vega-lite/docs/encoding.html#channels>`__ to sort by (e.g.,
        ``"x"``, ``"y"`` )
    order : None, :class:`SortOrder`, Literal['ascending', 'descending']
        The sort order. One of ``"ascending"`` (default), ``"descending"``, or ``null`` (no
        not sort).
    """
    _schema = {'$ref': '#/definitions/SortByEncoding'}

    def __init__(self, encoding: Union['SchemaBase', Literal['x', 'y', 'color', 'fill', 'stroke', 'strokeWidth', 'size', 'shape', 'fillOpacity', 'strokeOpacity', 'opacity', 'text'], UndefinedType]=Undefined, order: Union[None, 'SchemaBase', Literal['ascending', 'descending'], UndefinedType]=Undefined, **kwds):
        super(SortByEncoding, self).__init__(encoding=encoding, order=order, **kwds)