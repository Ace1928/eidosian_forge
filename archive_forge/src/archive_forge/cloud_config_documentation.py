from heat.common.i18n import _
from heat.common import template_format
from heat.engine import properties
from heat.engine.resources.openstack.heat import software_config
from heat.engine import support
from heat.rpc import api as rpc_api
A configuration resource for representing cloud-init cloud-config.

    This resource allows cloud-config YAML to be defined and stored by the
    config API. Any intrinsic functions called in the config will be resolved
    before storing the result.

    This resource will generally be referenced by OS::Nova::Server user_data,
    or OS::Heat::MultipartMime parts config. Since cloud-config is boot-only
    configuration, any changes to the definition will result in the
    replacement of all servers which reference it.
    