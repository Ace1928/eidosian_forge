from unittest import mock
from oslo_serialization import jsonutils
import sys
from keystoneauth1 import fixture
import requests
class FakeResource(object):

    def __init__(self, manager=None, info=None, loaded=False, methods=None):
        """Set attributes and methods for a resource.

        :param manager:
            The resource manager
        :param Dictionary info:
            A dictionary with all attributes
        :param bool loaded:
            True if the resource is loaded in memory
        :param Dictionary methods:
            A dictionary with all methods
        """
        info = info or {}
        methods = methods or {}
        self.__name__ = type(self).__name__
        self.manager = manager
        self._info = info
        self._add_details(info)
        self._add_methods(methods)
        self._loaded = loaded

    def _add_details(self, info):
        for k, v in info.items():
            setattr(self, k, v)

    def _add_methods(self, methods):
        """Fake methods with MagicMock objects.

        For each <@key, @value> pairs in methods, add an callable MagicMock
        object named @key as an attribute, and set the mock's return_value to
        @value. When users access the attribute with (), @value will be
        returned, which looks like a function call.
        """
        for name, ret in methods.items():
            method = mock.Mock(return_value=ret)
            setattr(self, name, method)

    def __repr__(self):
        reprkeys = sorted((k for k in self.__dict__.keys() if k[0] != '_' and k != 'manager'))
        info = ', '.join(('%s=%s' % (k, getattr(self, k)) for k in reprkeys))
        return '<%s %s>' % (self.__class__.__name__, info)

    def keys(self):
        return self._info.keys()

    def to_dict(self):
        return self._info

    @property
    def info(self):
        return self._info

    def __getitem__(self, item):
        return self._info.get(item)

    def get(self, item, default=None):
        return self._info.get(item, default)