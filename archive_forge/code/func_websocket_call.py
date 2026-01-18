from kubernetes.client.rest import ApiException
import select
import certifi
import time
import collections
from websocket import WebSocket, ABNF, enableTrace
import six
import ssl
from six.moves.urllib.parse import urlencode, quote_plus, urlparse, urlunparse
def websocket_call(configuration, *args, **kwargs):
    """An internal function to be called in api-client when a websocket

    connection is required. args and kwargs are the parameters of
    apiClient.request method.
  """
    url = args[1]
    _request_timeout = kwargs.get('_request_timeout', 60)
    _preload_content = kwargs.get('_preload_content', True)
    headers = kwargs.get('headers')
    query_params = []
    for key, value in kwargs.get('query_params', {}):
        if key == 'command' and isinstance(value, list):
            for command in value:
                query_params.append((key, command))
        else:
            query_params.append((key, value))
    if query_params:
        url += '?' + urlencode(query_params)
    try:
        client = WSClient(configuration, get_websocket_url(url), headers)
        if not _preload_content:
            return client
        client.run_forever(timeout=_request_timeout)
        return WSResponse('%s' % ''.join(client.read_all()))
    except (Exception, KeyboardInterrupt, SystemExit) as e:
        raise ApiException(status=0, reason=str(e))