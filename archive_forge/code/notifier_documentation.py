import collections
import contextlib
import copy
import logging
from oslo_utils import reflection
Checks if the event can be registered/subscribed to.

        :param event_type: event that needs to be verified
        :returns: whether the event can be registered/subscribed to
        :rtype: boolean
        