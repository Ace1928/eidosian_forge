import abc
import collections
import itertools
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import strutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import param_utils
from heat.engine import constraints as constr
@classmethod
def _value_as_text(cls, value):
    return encodeutils.safe_decode(jsonutils.dumps(value))