import datetime
from email.utils import quote
import io
import json
import random
import re
import sys
import time
from lazr.uri import URI, merge
from wadllib import (
from wadllib.iso_strptime import iso_strptime
def _get_definition_url(self):
    """Find the URL containing the definition ."""
    type = self.tag.attrib.get('resource_type')
    if type is None:
        raise WADLError('Parameter is a link, but not to a resource with a known WADL description.')
    return type