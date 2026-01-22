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
def _obj_make_obj_compatible(self, primitive, target_version, field):
    """Backlevel a sub-object based on our versioning rules.

        This is responsible for backporting objects contained within
        this object's primitive according to a set of rules we
        maintain about version dependencies between objects. This
        requires that the obj_relationships table in this object is
        correct and up-to-date.

        :param:primitive: The primitive version of this object
        :param:target_version: The version string requested for this object
        :param:field: The name of the field in this object containing the
                      sub-object to be backported
        """
    relationship_map = self._obj_relationship_for(field, target_version)
    if not relationship_map:
        return
    try:
        _get_subobject_version(target_version, relationship_map, lambda ver: _do_subobject_backport(ver, self, field, primitive))
    except exception.TargetBeforeSubobjectExistedException:
        del primitive[field]