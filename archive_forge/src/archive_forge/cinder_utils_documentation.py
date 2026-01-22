import logging
from cinderclient.apiclient import exceptions as apiclient_exception
from cinderclient import exceptions as cinder_exception
from keystoneauth1 import exceptions as keystone_exc
from oslo_utils import excutils
import retrying
from glance_store import exceptions
from glance_store.i18n import _LE
Extend volume

        :param client: cinderclient object
        :param volume: UUID of the volume to extend
        :param new_size: new size of the volume after extend
        