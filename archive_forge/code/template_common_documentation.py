import collections
import functools
import weakref
from heat.common import exception
from heat.common.i18n import _
from heat.engine import conditions
from heat.engine import function
from heat.engine import output
from heat.engine import template
Return the condition definitions of template.