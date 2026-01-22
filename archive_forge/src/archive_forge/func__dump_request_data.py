import collections
from requests import compat
def _dump_request_data(request, prefixes, bytearr, proxy_info=None):
    if proxy_info is None:
        proxy_info = {}
    prefix = prefixes.request
    method = _coerce_to_bytes(proxy_info.pop('method', request.method))
    request_path, uri = _build_request_path(request.url, proxy_info)
    bytearr.extend(prefix + method + b' ' + request_path + b' HTTP/1.1\r\n')
    headers = request.headers.copy()
    host_header = _coerce_to_bytes(headers.pop('Host', uri.netloc))
    bytearr.extend(prefix + b'Host: ' + host_header + b'\r\n')
    for name, value in headers.items():
        bytearr.extend(prefix + _format_header(name, value))
    bytearr.extend(prefix + b'\r\n')
    if request.body:
        if isinstance(request.body, compat.basestring):
            bytearr.extend(prefix + _coerce_to_bytes(request.body))
        else:
            bytearr.extend(b'<< Request body is not a string-like type >>')
        bytearr.extend(b'\r\n')
    bytearr.extend(b'\r\n')