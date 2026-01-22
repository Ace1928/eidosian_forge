from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
A resource implements Mistral cron trigger.

    Cron trigger is an object allowing to run workflow on a schedule. User
    specifies what workflow with what input needs to be run and also specifies
    how often it should be run. Pattern property is used to describe the
    frequency of workflow execution.
    