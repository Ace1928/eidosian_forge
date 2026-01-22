import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
class BaseMonitorEvent:

    def __init__(self, service, operation, timestamp):
        """Base monitor event

        :type service: str
        :param service: A string identifying the service associated to
            the event

        :type operation: str
        :param operation: A string identifying the operation of service
            associated to the event

        :type timestamp: int
        :param timestamp: Epoch time in milliseconds from when the event began
        """
        self.service = service
        self.operation = operation
        self.timestamp = timestamp

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False