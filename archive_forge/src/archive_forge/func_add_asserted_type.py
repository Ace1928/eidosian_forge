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
def add_asserted_type(self, type_identifier):
    """
        Adds a PROV type assertion to the record.

        :param type_identifier: PROV namespace identifier to add.
        """
    self._attributes[PROV_TYPE].add(type_identifier)