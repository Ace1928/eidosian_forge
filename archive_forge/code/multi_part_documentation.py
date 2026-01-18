import email
from email.mime import multipart
from email.mime import text
import os
from oslo_utils import uuidutils
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.heat import software_config
from heat.engine import support
from heat.rpc import api as rpc_api
Assembles a collection of software configurations as a multi-part mime.

    Parts in the message can be populated with inline configuration or
    references to other config resources. If the referenced resource is itself
    a valid multi-part mime message, that will be broken into parts and
    those parts appended to this message.

    The resulting multi-part mime message will be stored by the configs API
    and can be referenced in properties such as OS::Nova::Server user_data.

    This resource is generally used to build a list of cloud-init
    configuration elements including scripts and cloud-config. Since
    cloud-init is boot-only configuration, any changes to the definition
    will result in the replacement of all servers which reference it.
    