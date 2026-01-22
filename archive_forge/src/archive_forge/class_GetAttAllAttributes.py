import collections
import functools
import hashlib
import itertools
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from urllib import parse as urlparse
import yaql
from yaql.language import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import function
class GetAttAllAttributes(GetAtt):
    """A function for resolving resource attributes.

    Takes the form::

        get_attr:
          - <resource_name>
          - <attributes_name>
          - <path1>
          - ...

    where <attributes_name> and <path1>, ... are optional arguments. If there
    is no <attributes_name>, result will be dict of all resource's attributes.
    Else function returns resolved resource's attribute.
    """

    def _parse_args(self):
        if not self.args:
            raise ValueError(_('Arguments to "%s" can be of the next forms: [resource_name] or [resource_name, attribute, (path), ...]') % self.fn_name)
        elif isinstance(self.args, collections.abc.Sequence):
            if len(self.args) > 1:
                return super(GetAttAllAttributes, self)._parse_args()
            else:
                return (self.args[0], None, [])
        else:
            raise TypeError(_('Argument to "%s" must be a list') % self.fn_name)

    def _attr_path(self):
        if self._attribute is None:
            return attributes.ALL_ATTRIBUTES
        return super(GetAttAllAttributes, self)._attr_path()

    def result(self):
        if self._attribute is None:
            r = self._resource()
            if r.status in (r.IN_PROGRESS, r.COMPLETE) and r.action in (r.CREATE, r.ADOPT, r.SUSPEND, r.RESUME, r.UPDATE, r.CHECK, r.SNAPSHOT):
                return r.FnGetAtts()
            else:
                return None
        else:
            return super(GetAttAllAttributes, self).result()

    def _allow_without_attribute_name(self):
        return True