import contextlib
import logging
import os
import random
import sys
import time
from oslo_utils import reflection
from taskflow import engines
from taskflow.listeners import printing
from taskflow.patterns import unordered_flow as uf
from taskflow import task
class VolumeCreator(task.Task):

    def __init__(self, volume_id):
        base_name = reflection.get_callable_name(self)
        super(VolumeCreator, self).__init__(name='%s-%s' % (base_name, volume_id))
        self._volume_id = volume_id

    def execute(self):
        print('Making volume %s' % self._volume_id)
        time.sleep(random.random() * MAX_CREATE_TIME)
        print('Finished making volume %s' % self._volume_id)