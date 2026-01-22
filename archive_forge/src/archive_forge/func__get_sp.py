from oslo_log import log
from oslo_serialization import jsonutils
from sqlalchemy import orm
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.federation.backends import base
from keystone.i18n import _
def _get_sp(self, session, sp_id):
    sp_ref = session.get(ServiceProviderModel, sp_id)
    if not sp_ref:
        raise exception.ServiceProviderNotFound(sp_id=sp_id)
    return sp_ref