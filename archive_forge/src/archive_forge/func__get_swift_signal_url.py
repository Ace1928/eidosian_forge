from urllib import parse
from keystoneclient.contrib.ec2 import utils as ec2_utils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import password_gen
from heat.engine.clients.os import swift
from heat.engine.resources import stack_user
def _get_swift_signal_url(self, multiple_signals=False):
    """Create properly formatted and pre-signed Swift signal URL.

        This uses a Swift pre-signed temp_url. If multiple_signals is
        requested, the Swift object referenced by the returned URL will have
        versioning enabled.
        """
    put_url = self.data().get('swift_signal_url')
    if put_url:
        return put_url
    if self.id is None:
        return
    container = self.stack.id
    object_name = self.physical_resource_name()
    self.client('swift').put_container(container)
    if multiple_signals:
        put_url = self.client_plugin('swift').get_signal_url(container, object_name)
    else:
        put_url = self.client_plugin('swift').get_temp_url(container, object_name)
        self.client('swift').put_object(container, object_name, '')
    self.data_set('swift_signal_url', put_url)
    self.data_set('swift_signal_object_name', object_name)
    return put_url