import copy
import datetime
import jsonschema
import logging
import pytz
from unittest import mock
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import testtools
from testtools import matchers
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
@base.VersionedObjectRegistry.register
class MyCompoundObject(base.VersionedObject):
    fields = {'foo': fields.Field(fields.List(fields.Integer())), 'bar': fields.Field(fields.Dict(fields.Integer())), 'baz': fields.Field(fields.Set(fields.Integer()))}