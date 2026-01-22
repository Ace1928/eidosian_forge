from oslo_log import log
from oslo_serialization import jsonutils
from sqlalchemy import orm
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.federation.backends import base
from keystone.i18n import _
def _get_idp_from_remote_id(self, session, remote_id):
    q = session.query(IdPRemoteIdsModel)
    q = q.filter_by(remote_id=remote_id)
    try:
        return q.one()
    except sql.NotFound:
        raise exception.IdentityProviderNotFound(idp_id=remote_id)