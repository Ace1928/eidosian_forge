from openstack import exceptions
from openstack.instance_ha.v1 import host as _host
from openstack.instance_ha.v1 import notification as _notification
from openstack.instance_ha.v1 import segment as _segment
from openstack.instance_ha.v1 import vmove as _vmove
from openstack import proxy
from openstack import resource
def vmoves(self, notification, **query):
    """Return a generator of vmoves.

        :param notification: The value can be the UUID of a notification or
            a :class: `~masakariclient.sdk.ha.v1.notification.Notification`
            instance.
        :param kwargs query: Optional query parameters to be sent to
            limit the vmoves being returned.

        :returns: A generator of vmoves
        """
    notification_id = resource.Resource._get_id(notification)
    return self._list(_vmove.VMove, notification_id=notification_id, **query)