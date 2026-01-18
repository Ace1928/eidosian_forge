from __future__ import (absolute_import, division, print_function)
import atexit
import datetime
import itertools
import json
import os
import re
import ssl
import sys
import uuid
from time import time
from jinja2 import Environment
from ansible.module_utils.six import integer_types, PY3
from ansible.module_utils.six.moves import configparser
 Return hostvars for a single host 