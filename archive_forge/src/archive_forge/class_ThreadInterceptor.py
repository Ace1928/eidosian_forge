from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import re
import threading
import time
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.logging import common
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.credentials import requests as creds_requests
from googlecloudsdk.core.util import encoding
import requests
class ThreadInterceptor(threading.Thread):
    """Wrapper to intercept thread exceptions."""

    def __init__(self, target):
        super(ThreadInterceptor, self).__init__()
        self.target = target
        self.exception = None

    def run(self):
        try:
            self.target()
        except api_exceptions.HttpError as e:
            if e.status_code == 403:
                self.exception = DefaultLogsBucketIsOutsideSecurityPerimeterException()
            else:
                self.exception = e
        except api_exceptions.CommunicationError as e:
            self.exception = e