from collections import defaultdict
import datetime
import io
import json
from prov import Error
from prov.serializers import Serializer
from prov.constants import *
from prov.model import (
import logging
class ProvJSONDecoder(json.JSONDecoder):

    def decode(self, s, *args, **kwargs):
        container = super(ProvJSONDecoder, self).decode(s, *args, **kwargs)
        document = ProvDocument()
        decode_json_document(container, document)
        return document