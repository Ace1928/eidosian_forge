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
class AllocateIP(task.Task):
    """Allocates the ips for the given vm."""

    def __init__(self, name):
        super(AllocateIP, self).__init__(provides='ips', name=name)

    def execute(self, vm_spec):
        ips = []
        for _i in range(0, vm_spec.get('ips', 0)):
            ips.append('192.168.0.%s' % random.randint(1, 254))
        return ips