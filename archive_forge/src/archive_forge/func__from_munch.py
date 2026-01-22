import abc
import collections
import inspect
import itertools
import operator
import typing as ty
import urllib.parse
import warnings
import jsonpatch
from keystoneauth1 import adapter
from keystoneauth1 import discover
from requests import structures
from openstack import _log
from openstack import exceptions
from openstack import format
from openstack import utils
from openstack import warnings as os_warnings
@classmethod
def _from_munch(cls, obj, synchronized=True, connection=None):
    """Create an instance from a ``utils.Munch`` object.

        This is intended as a temporary measure to convert between shade-style
        Munch objects and original openstacksdk resources.

        :param obj: a ``utils.Munch`` object to convert from.
        :param bool synchronized: whether this object already exists on server
            Must be set to ``False`` for newly created objects.
        """
    return cls(_synchronized=synchronized, connection=connection, **obj)