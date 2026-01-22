import logging
import urllib
from copy import deepcopy
from mlflow.utils import rest_utils
from mlflow.utils.file_utils import read_chunk
def _append_query_parameters(url, parameters):
    parsed_url = urllib.parse.urlparse(url)
    query_dict = dict(urllib.parse.parse_qsl(parsed_url.query))
    query_dict.update(parameters)
    new_query = urllib.parse.urlencode(query_dict)
    new_url_components = parsed_url._replace(query=new_query)
    return urllib.parse.urlunparse(new_url_components)