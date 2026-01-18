from openstack import exceptions
from openstack import resource
from openstack import utils
Delete image from store

        :param session: The session to use for making this request.
        :param image: The value can be either the ID of an image or a
            :class:`~openstack.image.v2.image.Image` instance.

        :returns: The result of the ``delete`` if resource found, else None.
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when
            ignore_missing if ``False`` and a nonexistent resource
            is attempted to be deleted.
        