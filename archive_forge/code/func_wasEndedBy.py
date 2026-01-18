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