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
def _get_template_extension_manager():
    return extension.ExtensionManager(namespace='heat.templates', invoke_on_load=False, on_load_failure_callback=raise_extension_exception)