from openstack.dns.v2 import _base
from openstack import exceptions
from openstack import resource
Create a remote resource based on this instance.

        :param session: The session to use for making this request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param prepend_key: A boolean indicating whether the resource_key
                            should be prepended in a resource creation
                            request. Default to True.
        :param str base_path: Base part of the URI for creating resources, if
                              different from
                              :data:`~openstack.resource.Resource.base_path`.
        :return: This :class:`Resource` instance.
        :raises: :exc:`~openstack.exceptions.MethodNotSupported` if
                 :data:`Resource.allow_create` is not set to ``True``.
        