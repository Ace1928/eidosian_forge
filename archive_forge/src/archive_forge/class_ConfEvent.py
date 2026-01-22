import abc
from abc import ABCMeta
from abc import abstractmethod
import functools
import numbers
import logging
import uuid
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.base import get_validator
from os_ken.services.protocols.bgp.base import RUNTIME_CONF_ERROR_CODE
from os_ken.services.protocols.bgp.base import validate
from os_ken.services.protocols.bgp.utils import validation
from os_ken.services.protocols.bgp.utils.validation import is_valid_asn
@functools.total_ordering
class ConfEvent(object):
    """Encapsulates configuration settings change/update event."""

    def __init__(self, evt_src, evt_name, evt_value):
        """Creates an instance using given parameters.

        Parameters:
            -`evt_src`: (BaseConf) source of the event
            -`evt_name`: (str) name of event, has to be one of the valid
            event of `evt_src`
            - `evt_value`: (tuple) event context that helps event handler
        """
        if evt_name not in evt_src.get_valid_evts():
            raise ValueError('Event %s is not a valid event for type %s.' % (evt_name, type(evt_src)))
        self._src = evt_src
        self._name = evt_name
        self._value = evt_value

    @property
    def src(self):
        return self._src

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value

    def __repr__(self):
        return '<ConfEvent(%s, %s, %s)>' % (self.src, self.name, self.value)

    def __str__(self):
        return 'ConfEvent(src=%s, name=%s, value=%s)' % (self.src, self.name, self.value)

    def __lt__(self, other):
        return (self.src, self.name, self.value) < (other.src, other.name, other.value)

    def __eq__(self, other):
        return (self.src, self.name, self.value) == (other.src, other.name, other.value)