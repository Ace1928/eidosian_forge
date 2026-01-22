from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from typing import Mapping, Sequence
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.command_lib.run.printers import k8s_object_printer_util as k8s_util
from googlecloudsdk.core.resource import custom_printer_base as cp
def Containers():
    for name, container in k8s_util.OrderByKey(record.containers):
        key = 'Container {name}'.format(name=name)
        value = GetContainer(container, record.labels, dependencies[name], len(record.containers) == 1 or container.ports)
        yield (key, value)