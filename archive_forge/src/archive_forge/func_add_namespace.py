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
def add_namespace(self, namespace_or_prefix, uri=None):
    """
        Adds a namespace (if not available, yet).

        :param namespace_or_prefix: :py:class:`~prov.identifier.Namespace` or its
            prefix as a string to add.
        :param uri: Namespace URI (default: None). Must be present if only a
            prefix is given in the previous parameter.
        """
    if uri is None:
        return self._namespaces.add_namespace(namespace_or_prefix)
    else:
        return self._namespaces.add_namespace(Namespace(namespace_or_prefix, uri))