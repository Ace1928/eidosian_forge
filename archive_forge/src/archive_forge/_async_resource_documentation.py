from openstack.clustering.v1 import action as _action
from openstack import exceptions
from openstack import resource
Delete the remote resource based on this instance.

        :param session: The session to use for making this request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`

        :return: An :class:`~openstack.clustering.v1.action.Action`
                 instance. The ``fetch`` method will need to be used
                 to populate the `Action` with status information.
        :raises: :exc:`~openstack.exceptions.MethodNotSupported` if
                 :data:`Resource.allow_commit` is not set to ``True``.
        :raises: :exc:`~openstack.exceptions.ResourceNotFound` if
                 the resource was not found.
        