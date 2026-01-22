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
def _delete_swift_signal_url(self):
    object_name = self.data().get('swift_signal_object_name')
    if not object_name:
        return
    with self.client_plugin('swift').ignore_not_found:
        container_name = self.stack.id
        swift = self.client('swift')
        container = swift.get_container(container_name)
        filtered = [obj for obj in container[1] if object_name in obj['name']]
        for obj in filtered:
            swift.delete_object(container_name, object_name)
        headers = swift.head_container(container_name)
        if int(headers['x-container-object-count']) == 0:
            swift.delete_container(container_name)
    self.data_delete('swift_signal_object_name')
    self.data_delete('swift_signal_url')