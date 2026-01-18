import abc
import collections
from collections import abc as collections_abc
import copy
import functools
import logging
import warnings
import oslo_messaging as messaging
from oslo_utils import excutils
from oslo_utils import versionutils as vutils
from oslo_versionedobjects._i18n import _
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields as obj_fields
@classmethod
def to_json_schema(cls):
    obj_name = cls.obj_name()
    schema = {'$schema': 'http://json-schema.org/draft-04/schema#', 'title': obj_name}
    schema.update(obj_fields.Object(obj_name).get_schema())
    return schema