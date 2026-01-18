import os
import time
import typing as ty
import warnings
from openstack import exceptions
from openstack.image.v2 import cache as _cache
from openstack.image.v2 import image as _image
from openstack.image.v2 import member as _member
from openstack.image.v2 import metadef_namespace as _metadef_namespace
from openstack.image.v2 import metadef_object as _metadef_object
from openstack.image.v2 import metadef_property as _metadef_property
from openstack.image.v2 import metadef_resource_type as _metadef_resource_type
from openstack.image.v2 import metadef_schema as _metadef_schema
from openstack.image.v2 import schema as _schema
from openstack.image.v2 import service_info as _si
from openstack.image.v2 import task as _task
from openstack import proxy
from openstack import resource
from openstack import utils
from openstack import warnings as os_warnings
def update_member(self, member, image, **attrs):
    """Update the member of an image

        :param member: Either the ID of a member or a
            :class:`~openstack.image.v2.member.Member` instance.
        :param image: This is the image that the member belongs to.
            The value can be the ID of a image or a
            :class:`~openstack.image.v2.image.Image` instance.
        :param attrs: The attributes to update on the member represented
            by ``member``.

        See `Image Sharing Reference
        <https://docs.openstack.org/api-ref/image/v2/index.html?expanded=update-image-member-detail#update-image-member>`__
        for details.

        :returns: The updated member
        :rtype: :class:`~openstack.image.v2.member.Member`
        """
    member_id = resource.Resource._get_id(member)
    image_id = resource.Resource._get_id(image)
    return self._update(_member.Member, member_id=member_id, image_id=image_id, **attrs)