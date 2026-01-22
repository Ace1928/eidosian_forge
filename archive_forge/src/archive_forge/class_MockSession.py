from __future__ import absolute_import, division, print_function
import os
import sys
import copy
import json
import logging
import time
from datetime import datetime, timedelta
from ssl import SSLError
class MockSession(object):

    def __init__(self, *args, **kwargs):
        raise Exception('Requests library Session object not found. Using fake one.')