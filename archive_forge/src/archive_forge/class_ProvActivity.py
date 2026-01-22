from collections import defaultdict
from copy import deepcopy
import datetime
import io
import itertools
import logging
import os
import shutil
import tempfile
from urllib.parse import urlparse
import dateutil.parser
from prov import Error, serializers
from prov.constants import *
from prov.identifier import Identifier, QualifiedName, Namespace
class ProvActivity(ProvElement):
    """Provenance Activity element."""
    FORMAL_ATTRIBUTES = (PROV_ATTR_STARTTIME, PROV_ATTR_ENDTIME)
    _prov_type = PROV_ACTIVITY

    def set_time(self, startTime=None, endTime=None):
        """
        Sets the time this activity took place.

        :param startTime: Start time for the activity.
            Either a :py:class:`datetime.datetime` object or a string that can be
            parsed by :py:func:`dateutil.parser`.
        :param endTime: Start time for the activity.
            Either a :py:class:`datetime.datetime` object or a string that can be
            parsed by :py:func:`dateutil.parser`.
        """
        if startTime is not None:
            self._attributes[PROV_ATTR_STARTTIME] = {startTime}
        if endTime is not None:
            self._attributes[PROV_ATTR_ENDTIME] = {endTime}

    def get_startTime(self):
        """
        Returns the time the activity started.

        :return: :py:class:`datetime.datetime`
        """
        values = self._attributes[PROV_ATTR_STARTTIME]
        return first(values) if values else None

    def get_endTime(self):
        """
        Returns the time the activity ended.

        :return: :py:class:`datetime.datetime`
        """
        values = self._attributes[PROV_ATTR_ENDTIME]
        return first(values) if values else None

    def used(self, entity, time=None, attributes=None):
        """
        Creates a new usage record for this activity.

        :param entity: Entity or string identifier of the entity involved in
            the usage relationship (default: None).
        :param time: Optional time for the usage (default: None).
            Either a :py:class:`datetime.datetime` object or a string that can be
            parsed by :py:func:`dateutil.parser`.
        :param attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        self._bundle.usage(self, entity, time, other_attributes=attributes)
        return self

    def wasInformedBy(self, informant, attributes=None):
        """
        Creates a new communication record for this activity.

        :param informant: The informing activity (relationship source).
        :param attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        self._bundle.communication(self, informant, other_attributes=attributes)
        return self

    def wasStartedBy(self, trigger, starter=None, time=None, attributes=None):
        """
        Creates a new start record for this activity. The activity did not exist
        before the start by the trigger.

        :param trigger: Entity triggering the start of this activity.
        :param starter: Optionally extra activity to state a qualified start
            through which the trigger entity for the start is generated
            (default: None).
        :param time: Optional time for the start (default: None).
            Either a :py:class:`datetime.datetime` object or a string that can be
            parsed by :py:func:`dateutil.parser`.
        :param attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        self._bundle.start(self, trigger, starter, time, other_attributes=attributes)
        return self

    def wasEndedBy(self, trigger, ender=None, time=None, attributes=None):
        """
        Creates a new end record for this activity.

        :param trigger: Entity triggering the end of this activity.
        :param ender: Optionally extra activity to state a qualified end through
            which the trigger entity for the end is generated (default: None).
        :param time: Optional time for the end (default: None).
            Either a :py:class:`datetime.datetime` object or a string that can be
            parsed by :py:func:`dateutil.parser`.
        :param attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        self._bundle.end(self, trigger, ender, time, other_attributes=attributes)
        return self

    def wasAssociatedWith(self, agent, plan=None, attributes=None):
        """
        Creates a new association record for this activity.

        :param agent: Agent or string identifier of the agent involved in the
            association (default: None).
        :param plan: Optionally extra entity to state qualified association through
            an internal plan (default: None).
        :param attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
        self._bundle.association(self, agent, plan, other_attributes=attributes)
        return self