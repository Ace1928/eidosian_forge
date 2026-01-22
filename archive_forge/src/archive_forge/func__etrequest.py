import os
from packaging.version import Version
from .config import ET_PROJECTS
def _etrequest(endpoint, method='get', **kwargs):
    from requests import request, ConnectionError, ReadTimeout
    if kwargs.get('timeout') is None:
        kwargs['timeout'] = 5
    params = {}
    if ci_info.is_ci():
        params = ci_info.info()
    try:
        res = request(method, endpoint, params=params, **kwargs)
    except ConnectionError:
        raise RuntimeError('Connection to server could not be made')
    except ReadTimeout:
        raise RuntimeError('No response from server in {timeout} seconds'.format(timeout=kwargs.get('timeout')))
    res.raise_for_status()
    return res