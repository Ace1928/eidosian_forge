from heat.common.i18n import _
from heat.engine.resources.openstack.heat import none_resource
from heat.engine import support
class DesignateDomain(none_resource.NoneResource):
    """Heat Template Resource for Designate Domain.

    Designate provides DNS-as-a-Service services for OpenStack. So, domain
    is a realm with an identification string, unique in DNS.
    """
    support_status = support.SupportStatus(status=support.HIDDEN, version='10.0.0', message=_('This resource has been removed, use OS::Designate::Zone instead.'), previous_status=support.SupportStatus(status=support.DEPRECATED, version='8.0.0', previous_status=support.SupportStatus(version='5.0.0')))