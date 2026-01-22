from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ScaleBinParams(ScaleBins):
    """ScaleBinParams schema wrapper

    Parameters
    ----------

    step : float
        The step size defining the bin interval width.
    start : float
        The starting (lowest-valued) bin boundary.

        **Default value:** The lowest value of the scale domain will be used.
    stop : float
        The stopping (highest-valued) bin boundary.

        **Default value:** The highest value of the scale domain will be used.
    """
    _schema = {'$ref': '#/definitions/ScaleBinParams'}

    def __init__(self, step: Union[float, UndefinedType]=Undefined, start: Union[float, UndefinedType]=Undefined, stop: Union[float, UndefinedType]=Undefined, **kwds):
        super(ScaleBinParams, self).__init__(step=step, start=start, stop=stop, **kwds)