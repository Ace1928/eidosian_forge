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
class NoBoundRepresentationError(WADLError):
    """An unbound resource was used where wadllib expected a bound resource.

    To obtain the value of a resource's parameter, you first must bind
    the resource to a representation. Otherwise the resource has no
    idea what the value is and doesn't even know if you've given it a
    parameter name that makes sense.
    """