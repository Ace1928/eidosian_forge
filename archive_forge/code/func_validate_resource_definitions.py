import abc
import collections
import copy
import functools
import hashlib
from stevedore import extension
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine import conditions
from heat.engine import environment
from heat.engine import function
from heat.engine import template_files
from heat.objects import raw_template as template_object
def validate_resource_definitions(self, stack):
    """Check validity of resource definitions.

        This method is deprecated. Subclasses should validate the resource
        definitions in the process of generating them when calling
        resource_definitions(). However, for now this method is still called
        in case any third-party plugins are relying on this for validation and
        need time to migrate.
        """
    pass