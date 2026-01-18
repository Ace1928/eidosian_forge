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
def update_metadef_object(self, metadef_object, namespace, **attrs):
    """Update a single metadef object

        :param metadef_object: The value can be the ID of a metadef_object or a
            :class:`~openstack.image.v2.metadef_object.MetadefObject` instance.
        :param namespace: The value can be either the name of a metadef
            namespace or a
            :class:`~openstack.image.v2.metadef_namespace.MetadefNamespace`
            instance.
        :param dict attrs: Keyword arguments which will be used to update
            a :class:`~openstack.image.v2.metadef_object.MetadefObject`

        :returns: One :class:`~openstack.image.v2.metadef_object.MetadefObject`
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            resource can be found.
        """
    namespace_name = resource.Resource._get_id(namespace)
    return self._update(_metadef_object.MetadefObject, metadef_object, namespace_name=namespace_name, **attrs)