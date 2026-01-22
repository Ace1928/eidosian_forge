import sqlalchemy
from keystone.common import sql
from keystone.models import revoke_model
from keystone.revoke.backends import base
from oslo_db import api as oslo_db_api
def _list_token_events(self, token):
    with sql.session_for_read() as session:
        query = session.query(RevocationEvent).filter(RevocationEvent.issued_before >= token['issued_at'])
        user = [RevocationEvent.user_id.is_(None)]
        proj = [RevocationEvent.project_id.is_(None)]
        audit = [RevocationEvent.audit_id.is_(None)]
        trust = [RevocationEvent.trust_id.is_(None)]
        if token['user_id']:
            user.append(RevocationEvent.user_id == token['user_id'])
        if token['trustor_id']:
            user.append(RevocationEvent.user_id == token['trustor_id'])
        if token['trustee_id']:
            user.append(RevocationEvent.user_id == token['trustee_id'])
        if token['project_id']:
            proj.append(RevocationEvent.project_id == token['project_id'])
        if token['audit_id']:
            audit.append(RevocationEvent.audit_id == token['audit_id'])
        if token['trust_id']:
            trust.append(RevocationEvent.trust_id == token['trust_id'])
        query = query.filter(sqlalchemy.and_(sqlalchemy.or_(*user), sqlalchemy.or_(*proj), sqlalchemy.or_(*audit), sqlalchemy.or_(*trust)))
        events = [revoke_model.RevokeEvent(**e.to_dict()) for e in query]
        return events