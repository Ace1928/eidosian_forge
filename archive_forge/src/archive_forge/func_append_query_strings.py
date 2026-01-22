from urllib import parse
from troveclient.apiclient import exceptions
def append_query_strings(url, **query_strings):
    if not query_strings:
        return url
    query = '&'.join(('{0}={1}'.format(key, val) for key, val in query_strings.items() if val is not None))
    return url + ('?' + query if query else '')