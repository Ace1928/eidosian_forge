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
def _get_unused_prefix(self, original_prefix):
    if original_prefix not in self:
        return original_prefix
    count = 1
    while True:
        new_prefix = '_'.join((original_prefix, str(count)))
        if new_prefix in self:
            count += 1
        else:
            return new_prefix