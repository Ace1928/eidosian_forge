from __future__ import absolute_import, division, print_function
import os
import sys
import copy
import json
import logging
import time
from datetime import datetime, timedelta
from ssl import SSLError
def _update_session_last_used(self):
    if self.key in sessionDict:
        sessionDict[self.key]['last_used'] = datetime.utcnow()