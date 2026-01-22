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
def _process_iterable(self, context, action_fn, values):
    """Process an iterable, taking an action on each value.

        :param:context: Request context
        :param:action_fn: Action to take on each item in values
        :param:values: Iterable container of things to take action on
        :returns: A new container of the same type (except set) with
                  items from values having had action applied.
        """
    iterable = values.__class__
    if issubclass(iterable, dict):
        return iterable([(k, action_fn(context, v)) for k, v in values.items()])
    else:
        if iterable == set:
            iterable = list
        return iterable([action_fn(context, value) for value in values])