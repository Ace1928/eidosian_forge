import re
import xml.etree.ElementTree as etree
from io import BytesIO
from copy import deepcopy
from time import sleep
from base64 import b64encode
from typing import Dict
from functools import wraps
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class NttCisStatus:
    """
    NTTCIS API pending operation status class
        action, request_time, user_name, number_of_steps, update_time,
        step.name, step.number, step.percent_complete, failure_reason,
    """

    def __init__(self, action=None, request_time=None, user_name=None, number_of_steps=None, update_time=None, step_name=None, step_number=None, step_percent_complete=None, failure_reason=None):
        self.action = action
        self.request_time = request_time
        self.user_name = user_name
        self.number_of_steps = number_of_steps
        self.update_time = update_time
        self.step_name = step_name
        self.step_number = step_number
        self.step_percent_complete = step_percent_complete
        self.failure_reason = failure_reason

    def __repr__(self):
        return '<NttCisStatus: action=%s, request_time=%s, user_name=%s, number_of_steps=%s, update_time=%s, step_name=%s, step_number=%s, step_percent_complete=%s, failure_reason=%s>' % (self.action, self.request_time, self.user_name, self.number_of_steps, self.update_time, self.step_name, self.step_number, self.step_percent_complete, self.failure_reason)