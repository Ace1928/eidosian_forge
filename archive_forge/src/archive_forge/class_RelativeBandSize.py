from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class RelativeBandSize(VegaLiteSchema):
    """RelativeBandSize schema wrapper

    Parameters
    ----------

    band : float
        The relative band size.  For example ``0.5`` means half of the band scale's band
        width.
    """
    _schema = {'$ref': '#/definitions/RelativeBandSize'}

    def __init__(self, band: Union[float, UndefinedType]=Undefined, **kwds):
        super(RelativeBandSize, self).__init__(band=band, **kwds)