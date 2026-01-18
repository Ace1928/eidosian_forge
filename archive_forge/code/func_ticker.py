import http.client as http
import os
import re
import time
import psutil
import requests
from glance.tests import functional
from glance.tests.utils import execute
def ticker(self, message, seconds=60, tick=0.01):
    """
        Allows repeatedly testing for an expected result
        for a finite amount of time.

        :param message: Message to display on timeout
        :param seconds: Time in seconds after which we timeout
        :param tick: Time to sleep before rechecking for expected result
        :returns: 'True' or fails the test with 'message' on timeout
        """
    num_ticks = seconds * (1.0 / tick)
    count = 0
    while count < num_ticks:
        count += 1
        time.sleep(tick)
        yield
    self.fail(message)