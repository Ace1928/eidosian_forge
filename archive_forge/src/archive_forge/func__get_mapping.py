from oslo_log import log
from oslo_serialization import jsonutils
from sqlalchemy import orm
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.federation.backends import base
from keystone.i18n import _
def _get_mapping(self, session, mapping_id):
    mapping_ref = session.get(MappingModel, mapping_id)
    if not mapping_ref:
        raise exception.MappingNotFound(mapping_id=mapping_id)
    return mapping_ref