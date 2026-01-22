import datetime
import random as _random
import uuid
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystone.common import sql
from keystone.common import utils
from keystone import exception
from keystone.i18n import _
from keystone.oauth1.backends import base
def _delete_request_tokens(self, session, consumer_id):
    q = session.query(RequestToken)
    req_tokens = q.filter_by(consumer_id=consumer_id)
    req_tokens_list = set([x.id for x in req_tokens])
    for token_id in req_tokens_list:
        token_ref = self._get_request_token(session, token_id)
        session.delete(token_ref)