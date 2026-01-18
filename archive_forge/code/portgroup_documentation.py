from ironicclient.common import base
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
Update the Portgroup.

        :param portgroup_id: The UUID or name of a portgroup.
        :param patch: The patch request with updates.
        :param os_ironic_api_version: String version (e.g. "1.35") to use for
            the request.  If not specified, the client's default is used.
        :param global_request_id: String containing global request ID header
            value (in form "req-<UUID>") to use for the request.
        