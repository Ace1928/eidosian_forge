import webbrowser
from urllib.parse import urlencode
def _generate_url(func, stable):
    """
    Parse inputs and return a correctly formatted URL or raises ValueError
    if the input is not understandable
    """
    url = BASE_URL
    if stable:
        url += 'stable/'
    else:
        url += 'devel/'
    if func is None:
        return url
    elif isinstance(func, str):
        url += 'search.html?'
        url += urlencode({'q': func})
        url += '&check_keywords=yes&area=default'
    else:
        try:
            func = func
            func_name = func.__name__
            func_module = func.__module__
            if not func_module.startswith('statsmodels.'):
                raise ValueError('Function must be from statsmodels')
            url += 'generated/'
            url += func_module + '.' + func_name + '.html'
        except AttributeError:
            raise ValueError('Input not understood')
    return url