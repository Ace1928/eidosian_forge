import datetime
import logging
from lxml import etree
import io
import warnings
import prov
import prov.identifier
from prov.model import DEFAULT_NAMESPACES, sorted_attributes
from prov.constants import *  # NOQA
from prov.serializers import Serializer

        Helper function trying to derive the record label taking care of
        subtypes and what not. It will also remove the type declaration for
        the attributes if it was used to specialize the type.

        :param rec_type: The type of records.
        :param attributes: The attributes of the record.
        