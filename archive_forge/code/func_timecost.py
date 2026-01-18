from collections import abc
import decimal
import random
import weakref
from oslo_log import log as logging
from oslo_utils import timeutils
from oslo_utils import uuidutils
from neutron_lib._i18n import _
def timecost(f):
    call_id = uuidutils.generate_uuid()
    message_base = 'Time-cost: call %(call_id)s function %(fname)s ' % {'call_id': call_id, 'fname': f.__name__}
    end_message = message_base + 'took %(seconds).3fs seconds to run'

    @timeutils.time_it(LOG, message=end_message, min_duration=None)
    def wrapper(*args, **kwargs):
        LOG.debug(message_base + 'start')
        ret = f(*args, **kwargs)
        return ret
    return wrapper