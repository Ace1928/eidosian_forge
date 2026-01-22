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
def _delete_consumer(self, session, consumer_id):
    consumer_ref = self._get_consumer(session, consumer_id)
    session.delete(consumer_ref)