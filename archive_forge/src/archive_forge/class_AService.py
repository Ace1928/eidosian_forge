from unittest import mock
from testtools import matchers
from oslo_service import periodic_task
from oslo_service.tests import base
class AService(periodic_task.PeriodicTasks):

    def __init__(self, conf):
        super(AService, self).__init__(conf)

    @periodic_task.periodic_task(name='better-name')
    def tick(self, context):
        pass

    @periodic_task.periodic_task
    def tack(self, context):
        pass