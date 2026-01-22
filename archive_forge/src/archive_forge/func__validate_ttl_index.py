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
def _validate_ttl_index(self, collection, coll_name, ttl_seconds):
    """Checks if existing TTL index is removed on a collection.

        This logs warning when existing collection has TTL index defined and
        new cache configuration tries to disable index with
        ``mongo_ttl_seconds < 0``. In that case, existing index needs
        to be addressed first to make new configuration effective.
        Refer to MongoDB documentation around TTL index for further details.
        """
    indexes = collection.index_information()
    for indx_name, index_data in indexes.items():
        if all((k in index_data for k in ('key', 'expireAfterSeconds'))):
            existing_value = index_data['expireAfterSeconds']
            fld_present = 'doc_date' in index_data['key'][0]
            if fld_present and existing_value > -1 and (ttl_seconds < 1):
                msg = 'TTL index already exists on db collection <%(c_name)s>, remove index <%(indx_name)s> first to make updated mongo_ttl_seconds value to be  effective'
                LOG.warning(msg, {'c_name': coll_name, 'indx_name': indx_name})