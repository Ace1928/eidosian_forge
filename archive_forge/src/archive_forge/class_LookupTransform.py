from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class LookupTransform(Transform):
    """LookupTransform schema wrapper

    Parameters
    ----------

    lookup : str
        Key in primary data source.
    default : Any
        The default value to use if lookup fails.

        **Default value:** ``null``
    as : str, :class:`FieldName`, Sequence[str, :class:`FieldName`]
        The output fields on which to store the looked up data values.

        For data lookups, this property may be left blank if ``from.fields`` has been
        specified (those field names will be used); if ``from.fields`` has not been
        specified, ``as`` must be a string.

        For selection lookups, this property is optional: if unspecified, looked up values
        will be stored under a property named for the selection; and if specified, it must
        correspond to ``from.fields``.
    from : dict, :class:`LookupData`, :class:`LookupSelection`
        Data source or selection for secondary data reference.
    """
    _schema = {'$ref': '#/definitions/LookupTransform'}

    def __init__(self, lookup: Union[str, UndefinedType]=Undefined, default: Union[Any, UndefinedType]=Undefined, **kwds):
        super(LookupTransform, self).__init__(lookup=lookup, default=default, **kwds)