from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def _order_url_query_str(self, url):
    """Returns the url with the query strings ordered, if they exist and
        there's more than one. Otherwise the url is returned unaltered.
        """
    if self.URL_QUERY_SEPARATOR in url:
        parts = url.split(self.URL_SEPARATOR)
        if len(parts) == 2:
            queries = sorted(parts[1].split(self.URL_QUERY_SEPARATOR))
            url = self.URL_SEPARATOR.join([parts[0], self.URL_QUERY_SEPARATOR.join(queries)])
    return url