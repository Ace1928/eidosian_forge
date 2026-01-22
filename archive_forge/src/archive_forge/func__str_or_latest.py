import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def _str_or_latest(val):
    """Convert val to a string, handling LATEST => 'latest'.

    :param val: An int or the special value LATEST.
    :return: A string representation of val.  If val was LATEST, the return is
             'latest'.
    """
    return 'latest' if val == LATEST else str(val)