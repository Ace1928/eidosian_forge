import contextlib
import hashlib
import logging
import os
import random
import sys
import time
import futurist
from oslo_utils import uuidutils
from taskflow import engines
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import models
from taskflow import task
import example_utils as eu  # noqa
class AllocateVolumes(task.Task):
    """Allocates the volumes for the vm."""

    def execute(self, vm_spec):
        volumes = []
        for i in range(0, vm_spec['volumes']):
            with slow_down(1):
                volumes.append('/dev/vda%s' % (i + 1))
                print('Allocated volume %s' % volumes[-1])
        return volumes