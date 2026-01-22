from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ImputeSequence(VegaLiteSchema):
    """ImputeSequence schema wrapper

    Parameters
    ----------

    stop : float
        The ending value(exclusive) of the sequence.
    start : float
        The starting value of the sequence. **Default value:** ``0``
    step : float
        The step value between sequence entries. **Default value:** ``1`` or ``-1`` if
        ``stop < start``
    """
    _schema = {'$ref': '#/definitions/ImputeSequence'}

    def __init__(self, stop: Union[float, UndefinedType]=Undefined, start: Union[float, UndefinedType]=Undefined, step: Union[float, UndefinedType]=Undefined, **kwds):
        super(ImputeSequence, self).__init__(stop=stop, start=start, step=step, **kwds)