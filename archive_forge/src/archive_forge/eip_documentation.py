from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import client_exception
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.aws.ec2 import internet_gateway
from heat.engine.resources.aws.ec2 import vpc
from heat.engine import support
Remove a floating IP address from a server or port.