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
class ComparableVersionedObject(object):
    """Mix-in to provide comparison methods

    When objects are to be compared with each other (in tests for example),
    this mixin can be used.
    """

    def __eq__(self, obj):
        if hasattr(obj, 'obj_to_primitive'):
            return self.obj_to_primitive() == obj.obj_to_primitive()
        return NotImplemented

    def __hash__(self):
        return super(ComparableVersionedObject, self).__hash__()

    def __ne__(self, obj):
        if hasattr(obj, 'obj_to_primitive'):
            return self.obj_to_primitive() != obj.obj_to_primitive()
        return NotImplemented