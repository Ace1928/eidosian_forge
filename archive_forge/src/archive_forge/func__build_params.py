import copy
from oslo_utils import encodeutils
from oslo_utils import strutils
import urllib.parse
from glanceclient.common import utils
from glanceclient.v1.apiclient import base
def _build_params(self, parameters):
    params = {'limit': parameters.get('page_size', DEFAULT_PAGE_SIZE)}
    if 'marker' in parameters:
        params['marker'] = parameters['marker']
    sort_key = parameters.get('sort_key')
    if sort_key is not None:
        if sort_key in SORT_KEY_VALUES:
            params['sort_key'] = sort_key
        else:
            raise ValueError('sort_key must be one of the following: %s.' % ', '.join(SORT_KEY_VALUES))
    sort_dir = parameters.get('sort_dir')
    if sort_dir is not None:
        if sort_dir in SORT_DIR_VALUES:
            params['sort_dir'] = sort_dir
        else:
            raise ValueError('sort_dir must be one of the following: %s.' % ', '.join(SORT_DIR_VALUES))
    filters = parameters.get('filters', {})
    properties = filters.pop('properties', {})
    for key, value in properties.items():
        params['property-%s' % key] = value
    params.update(filters)
    if parameters.get('owner') is not None:
        params['is_public'] = None
    if 'is_public' in parameters:
        params['is_public'] = parameters['is_public']
    return params