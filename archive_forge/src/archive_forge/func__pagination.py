import inspect
import itertools
import logging
import re
import time
import urllib.parse as urlparse
import debtcollector.renames
from keystoneauth1 import exceptions as ksa_exc
import requests
from neutronclient._i18n import _
from neutronclient import client
from neutronclient.common import exceptions
from neutronclient.common import extension as client_extension
from neutronclient.common import serializer
from neutronclient.common import utils
def _pagination(self, collection, path, **params):
    if params.get('page_reverse', False):
        linkrel = 'previous'
    else:
        linkrel = 'next'
    next = True
    while next:
        res = self.get(path, params=params)
        yield res
        next = False
        try:
            for link in res['%s_links' % collection]:
                if link['rel'] == linkrel:
                    query_str = urlparse.urlparse(link['href']).query
                    params = urlparse.parse_qs(query_str)
                    next = True
                    break
        except KeyError:
            break