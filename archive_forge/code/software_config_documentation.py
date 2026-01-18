from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import properties
from heat.engine import resource
from heat.engine import software_config_io as swc_io
from heat.engine import support
from heat.rpc import api as rpc_api
Retrieve attributes of the SoftwareConfig resource.

        "config" returns the config value of the software config. If the
         software config does not exist, returns an empty string.
        