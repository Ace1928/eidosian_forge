from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class AutoSizeParams(VegaLiteSchema):
    """AutoSizeParams schema wrapper

    Parameters
    ----------

    contains : Literal['content', 'padding']
        Determines how size calculation should be performed, one of ``"content"`` or
        ``"padding"``. The default setting ( ``"content"`` ) interprets the width and height
        settings as the data rectangle (plotting) dimensions, to which padding is then
        added. In contrast, the ``"padding"`` setting includes the padding within the view
        size calculations, such that the width and height settings indicate the **total**
        intended size of the view.

        **Default value** : ``"content"``
    resize : bool
        A boolean flag indicating if autosize layout should be re-calculated on every view
        update.

        **Default value** : ``false``
    type : :class:`AutosizeType`, Literal['pad', 'none', 'fit', 'fit-x', 'fit-y']
        The sizing format type. One of ``"pad"``, ``"fit"``, ``"fit-x"``, ``"fit-y"``,  or
        ``"none"``. See the `autosize type
        <https://vega.github.io/vega-lite/docs/size.html#autosize>`__ documentation for
        descriptions of each.

        **Default value** : ``"pad"``
    """
    _schema = {'$ref': '#/definitions/AutoSizeParams'}

    def __init__(self, contains: Union[Literal['content', 'padding'], UndefinedType]=Undefined, resize: Union[bool, UndefinedType]=Undefined, type: Union['SchemaBase', Literal['pad', 'none', 'fit', 'fit-x', 'fit-y'], UndefinedType]=Undefined, **kwds):
        super(AutoSizeParams, self).__init__(contains=contains, resize=resize, type=type, **kwds)