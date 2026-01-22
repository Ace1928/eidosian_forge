import collections
import functools
import inspect
import socket
import flask
from oslo_log import log
import oslo_messaging
from oslo_utils import reflection
import pycadf
from pycadf import cadftaxonomy as taxonomy
from pycadf import cadftype
from pycadf import credential
from pycadf import eventfactory
from pycadf import host
from pycadf import reason
from pycadf import resource
from keystone.common import context
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
@classmethod
def added_to(cls, target_type, target_id, actor_type, actor_id, initiator=None, public=True, reason=None):
    actor_dict = {'id': actor_id, 'type': actor_type, 'actor_operation': 'added'}
    cls._emit(ACTIONS.updated, target_type, target_id, initiator, public, actor_dict=actor_dict, reason=reason)