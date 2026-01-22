import abc
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
import oslo_messaging
from oslo_utils import encodeutils
from oslo_utils import excutils
import webob
from glance.common import exception
from glance.common import timeutils
from glance.domain import proxy as domain_proxy
from glance.i18n import _, _LE
class TaskProxy(NotificationProxy, domain_proxy.Task):

    def get_super_class(self):
        return domain_proxy.Task

    def get_payload(self, obj):
        return format_task_notification(obj)

    def begin_processing(self):
        super(TaskProxy, self).begin_processing()
        self.send_notification('task.processing', self.repo)

    def succeed(self, result):
        super(TaskProxy, self).succeed(result)
        self.send_notification('task.success', self.repo)

    def fail(self, message):
        super(TaskProxy, self).fail(message)
        self.send_notification('task.failure', self.repo)

    def run(self, executor):
        super(TaskProxy, self).run(executor)
        self.send_notification('task.run', self.repo)