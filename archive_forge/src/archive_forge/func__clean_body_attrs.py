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
def _clean_body_attrs(self, attrs):
    """Mark the attributes as up-to-date."""
    self._body.clean(only=attrs)
    if self.commit_jsonpatch or self.allow_patch:
        for attr in attrs:
            if attr in self._body:
                self._original_body[attr] = self._body[attr]