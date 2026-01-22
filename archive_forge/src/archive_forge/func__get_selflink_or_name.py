import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _get_selflink_or_name(self, obj, get_selflinks=True, objname=None):
    """
        Return the selflink or name, given a name or object.

        Will try to fetch the appropriate object if necessary (assumes
        we only need one parameter to fetch the object, no introspection
        is performed).

        :param    obj: object to test.
        :type     obj: ``str`` or ``object``

        :param    get_selflinks: Inform if we should return selfLinks or just
                              the name.  Default is True.
        :param    get_selflinks: ``bool``

        :param    objname: string to use in constructing method call
        :type     objname: ``str`` or None

        :return:  URL from extra['selfLink'] or name
        :rtype:   ``str``
        """
    if get_selflinks:
        if not hasattr(obj, 'name'):
            if objname:
                getobj = getattr(self, 'ex_get_%s' % objname)
                obj = getobj(obj)
            else:
                raise ValueError('objname must be set if selflinks is True.')
        return obj.extra['selfLink']
    elif not hasattr(obj, 'name'):
        return obj
    else:
        return obj.name