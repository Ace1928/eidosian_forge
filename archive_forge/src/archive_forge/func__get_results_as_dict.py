import abc
import datetime
from dogpile.cache import api
from dogpile import util as dp_util
from oslo_cache import core
from oslo_log import log
from oslo_utils import importutils
from oslo_utils import timeutils
from oslo_cache._i18n import _
from oslo_cache import exception
def _get_results_as_dict(self, keys):
    criteria = {'_id': {'$in': keys}}
    db_results = self.get_cache_collection().find(spec=criteria, **self.meth_kwargs)
    return {doc['_id']: doc for doc in db_results}