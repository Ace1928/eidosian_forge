from collections import defaultdict
import datetime
import io
import json
from prov import Error
from prov.serializers import Serializer
from prov.constants import *
from prov.model import (
import logging

        Deserialize from the `PROV JSON
        <https://openprovenance.org/prov-json/>`_ representation to a
        :class:`~prov.model.ProvDocument` instance.

        :param stream: Input data.
        