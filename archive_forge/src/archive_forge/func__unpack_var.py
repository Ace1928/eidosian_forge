import os
import re
from copy import deepcopy
import numpy as np
from .arrayproxy import ArrayProxy
from .fileslice import strided_scalar
from .spatialimages import HeaderDataError, ImageDataError, SpatialHeader, SpatialImage
from .volumeutils import Recoder
def _unpack_var(var):
    """
    Parses key : value pair from `var`

    Parameters
    ----------
    var : str
        Entry from HEAD file

    Returns
    -------
    name : str
        Name of attribute
    value : object
        Value of attribute

    Examples
    --------
    >>> var = "type = integer-attribute\\nname = BRICK_TYPES\\ncount = 1\\n1\\n"
    >>> name, attr = _unpack_var(var)
    >>> print(name, attr)
    BRICK_TYPES 1
    >>> var = "type = string-attribute\\nname = TEMPLATE_SPACE\\ncount = 5\\n'ORIG~"
    >>> name, attr = _unpack_var(var)
    >>> print(name, attr)
    TEMPLATE_SPACE ORIG
    """
    err_msg = f'Please check HEAD file to ensure it is AFNI compliant. Offending attribute:\n{var}'
    atype, aname = (TYPE_RE.findall(var), NAME_RE.findall(var))
    if len(atype) != 1:
        raise AFNIHeaderError(f'Invalid attribute type entry in HEAD file. {err_msg}')
    if len(aname) != 1:
        raise AFNIHeaderError(f'Invalid attribute name entry in HEAD file. {err_msg}')
    atype = _attr_dic.get(atype[0], str)
    attr = ' '.join(var.strip().splitlines()[3:])
    if atype is not str:
        try:
            attr = [atype(f) for f in attr.split()]
        except ValueError:
            raise AFNIHeaderError(f'Failed to read variable from HEAD file due to improper type casting. {err_msg}')
    else:
        attr = attr.replace("'", '', 1).rstrip('~')
    return (aname[0], attr[0] if len(attr) == 1 else attr)