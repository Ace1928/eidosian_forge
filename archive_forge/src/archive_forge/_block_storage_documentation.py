import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
Delete volume quotas for a project

        :param name_or_id: project name or id

        :returns: The deleted volume ``QuotaSet`` object.
        :raises: :class:`~openstack.exceptions.SDKException` if it's not a
            valid project or the call failed
        