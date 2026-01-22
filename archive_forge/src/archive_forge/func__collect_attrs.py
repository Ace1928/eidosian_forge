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
def _collect_attrs(self, attrs):
    """Given attributes, return a dict per type of attribute

        This method splits up **attrs into separate dictionaries
        that correspond to the relevant body, header, and uri
        attributes that exist on this class.
        """
    body = self._consume_body_attrs(attrs)
    header = self._consume_header_attrs(attrs)
    uri = self._consume_uri_attrs(attrs)
    if attrs:
        if self._allow_unknown_attrs_in_body:
            body.update(attrs)
        elif self._store_unknown_attrs_as_properties:
            body = self._pack_attrs_under_properties(body, attrs)
    if any([body, header, uri]):
        attrs = self._compute_attributes(body, header, uri)
        body.update(self._consume_attrs(self._body_mapping(), attrs))
        header.update(self._consume_attrs(self._header_mapping(), attrs))
        uri.update(self._consume_attrs(self._uri_mapping(), attrs))
    computed = self._consume_attrs(self._computed_mapping(), attrs)
    if self._connection:
        computed.setdefault('location', self._connection.current_location)
    return (body, header, uri, computed)