from webob import exc
from neutron_lib._i18n import _
from neutron_lib.api.definitions import network
from neutron_lib.api.definitions import port
from neutron_lib.api.definitions import subnet
from neutron_lib.api.definitions import subnetpool
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions
def verify_attributes(self, attrs_to_verify):
    """Reject unknown attributes.

        Consumers should ensure the project info is populated in the
        attrs_to_verify before calling this method.

        :param attrs_to_verify: The attributes to verify against this
            resource attributes.
        :raises: HTTPBadRequest: If attrs_to_verify contains any unrecognized
            for this resource attributes instance.
        """
    extra_keys = set(attrs_to_verify.keys()) - set(self.attributes.keys())
    if extra_keys:
        msg = _("Unrecognized attribute(s) '%s'") % ', '.join(extra_keys)
        raise exc.HTTPBadRequest(msg)