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
def _get_subobject_version(tgt_version, relationships, backport_func):
    """Get the version to which we need to convert a subobject.

    This uses the relationships between a parent and a subobject,
    along with the target parent version, to decide the version we need
    to convert a subobject to. If the subobject did not exist in the parent at
    the target version, TargetBeforeChildExistedException is raised. If there
    is a need to backport, backport_func is called and the subobject version
    to backport to is passed in.

    :param tgt_version: The version we are converting the parent to
    :param relationships: A list of (parent, subobject) version tuples
    :param backport_func: A backport function that takes in the subobject
                          version
    :returns: The version we need to convert the subobject to
    """
    tgt = vutils.convert_version_to_tuple(tgt_version)
    for index, versions in enumerate(relationships):
        parent, child = versions
        parent = vutils.convert_version_to_tuple(parent)
        if tgt < parent:
            if index == 0:
                raise exception.TargetBeforeSubobjectExistedException(target_version=tgt_version)
            else:
                child = relationships[index - 1][1]
                backport_func(child)
            return
        elif tgt == parent:
            backport_func(child)
            return