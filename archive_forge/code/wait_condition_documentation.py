from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import wait_condition as wc_base
from heat.engine import support
Resource for handling signals received by WaitConditionHandle.

    Resource takes WaitConditionHandle and starts to create. Resource is in
    CREATE_IN_PROGRESS status until WaitConditionHandle doesn't receive
    sufficient number of successful signals (this number can be specified with
    count property) and successfully creates after that, or fails due to
    timeout.
    