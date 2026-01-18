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
def wasAttributedTo(self, agent, attributes=None):
    """
        Creates a new attribution record between this entity and an agent.

        :param agent: Agent or string identifier of the agent involved in the
            attribution.
        :param attributes: Optional other attributes as a dictionary or list
            of tuples to be added to the record optionally (default: None).
        """
    self._bundle.attribution(self, agent, other_attributes=attributes)
    return self