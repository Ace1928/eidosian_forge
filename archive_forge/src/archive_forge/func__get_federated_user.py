import copy
import datetime
import sqlalchemy
from oslo_config import cfg
from oslo_db import api as oslo_db_api
from keystone.common import provider_api
from keystone.common import sql
from keystone import exception
from keystone.identity.backends import base as identity_base
from keystone.identity.backends import sql_model as model
from keystone.identity.shadow_backends import base
def _get_federated_user(self, idp_id, protocol_id, unique_id):
    """Return the found user for the federated identity.

        :param idp_id: The identity provider ID
        :param protocol_id: The federation protocol ID
        :param unique_id: The user's unique ID (unique within the IdP)
        :returns User: Returns a reference to the User

        """
    with sql.session_for_read() as session:
        query = session.query(model.User).outerjoin(model.LocalUser)
        query = query.join(model.FederatedUser)
        query = query.filter(model.FederatedUser.idp_id == idp_id)
        query = query.filter(model.FederatedUser.protocol_id == protocol_id)
        query = query.filter(model.FederatedUser.unique_id == unique_id)
        try:
            user_ref = query.one()
        except sql.NotFound:
            raise exception.UserNotFound(user_id=unique_id)
        return user_ref