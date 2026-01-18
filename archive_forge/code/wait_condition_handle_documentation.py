from oslo_serialization import jsonutils
from heat.common.i18n import _
from heat.common import password_gen
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.aws.cfn import wait_condition_handle as aws_wch
from heat.engine.resources import signal_responder
from heat.engine.resources import wait_condition as wc_base
from heat.engine import support
Validate and update the resource metadata.

        Metadata is not mandatory, but if passed it must use the following
        format::

            {
                "status" : "Status (must be SUCCESS or FAILURE)",
                "data" : "Arbitrary data",
                "reason" : "Reason string"
            }

        Optionally "id" may also be specified, but if missing the index
        of the signal received will be used.
        