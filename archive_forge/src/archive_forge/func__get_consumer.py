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
def _get_consumer(self, session, consumer_id):
    consumer_ref = session.get(Consumer, consumer_id)
    if consumer_ref is None:
        raise exception.NotFound(_('Consumer not found'))
    return consumer_ref