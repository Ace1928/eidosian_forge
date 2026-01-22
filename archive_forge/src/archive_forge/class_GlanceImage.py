from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.heat import none_resource
from heat.engine import support
class GlanceImage(none_resource.NoneResource):
    """A resource managing images in Glance.

    A resource provides managing images that are meant to be used with other
    services.
    """
    support_status = support.SupportStatus(status=support.HIDDEN, version='22.0.0', previous_status=support.SupportStatus(status=support.DEPRECATED, version='8.0.0', message=_('Creating a Glance Image based on an existing URL location requires the Glance v1 API, which is deprecated.'), previous_status=support.SupportStatus(version='2014.2')))