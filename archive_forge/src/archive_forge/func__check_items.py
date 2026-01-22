import copy
import urllib
from oslo_serialization import jsonutils
from keystoneauth1 import exceptions
from mistralclient import utils
def _check_items(obj, searches):
    try:
        return all((getattr(obj, attr) == value for attr, value in searches))
    except AttributeError:
        return False