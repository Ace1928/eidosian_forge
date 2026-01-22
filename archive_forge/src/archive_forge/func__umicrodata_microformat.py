import copy
from typing import Any
from six.moves.urllib.parse import urljoin, urlparse
from extruct.dublincore import get_lower_attrib
def _umicrodata_microformat(extracted, schema_context):
    res = []
    if isinstance(extracted, list):
        for obj in extracted:
            res.append(flatten_dict(obj, schema_context, True))
    elif isinstance(extracted, dict):
        res.append(flatten_dict(extracted, schema_context, False))
    return res