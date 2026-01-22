import datetime
import os
import jwt
from oslo_utils import timeutils
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.token.providers import base
def _convert_time_int_to_string(self, time_int):
    time_object = datetime.datetime.utcfromtimestamp(time_int)
    return utils.isotime(at=time_object, subsecond=True)