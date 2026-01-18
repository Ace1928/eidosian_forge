from __future__ import absolute_import, division, print_function
import os
import sys
import copy
import json
import logging
import time
from datetime import datetime, timedelta
from ssl import SSLError
@staticmethod
def to_avi_response(resp):
    if type(resp) is Response:
        return ApiResponse(resp)
    return resp