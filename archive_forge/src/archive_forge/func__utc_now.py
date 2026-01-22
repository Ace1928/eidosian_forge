import argparse
import datetime
import logging
import re
from oslo_serialization import jsonutils
from oslo_utils import strutils
from blazarclient import command
from blazarclient import exception
def _utc_now():
    """Wrap datetime.datetime.utcnow so it can be mocked in unit tests.

    This is required because some of the tests require understanding the
    'current time'; simply mocking utcnow() is made very difficult by
    the many different ways the datetime package is used in this module.
    """
    return datetime.datetime.utcnow()