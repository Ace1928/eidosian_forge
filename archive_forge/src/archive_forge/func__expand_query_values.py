from .._compat import basestring
from .._compat import urlencode as _urlencode
def _expand_query_values(original_query_list):
    query_list = []
    for key, value in original_query_list:
        if isinstance(value, basestring):
            query_list.append((key, value))
        else:
            key_fmt = key + '[%s]'
            value_list = _to_kv_list(value)
            query_list.extend(((key_fmt % k, v) for k, v in value_list))
    return query_list