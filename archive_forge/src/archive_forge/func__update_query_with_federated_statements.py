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
def _update_query_with_federated_statements(self, hints, query):
    statements = []
    for filter_ in hints.filters:
        if filter_['name'] == 'idp_id':
            statements.append(model.FederatedUser.idp_id == filter_['value'])
        if filter_['name'] == 'protocol_id':
            statements.append(model.FederatedUser.protocol_id == filter_['value'])
        if filter_['name'] == 'unique_id':
            statements.append(model.FederatedUser.unique_id == filter_['value'])
    hints.filters = [x for x in hints.filters if x['name'] not in ('idp_id', 'protocol_id', 'unique_id')]
    if statements:
        query = query.filter(sqlalchemy.and_(*statements))
    return query