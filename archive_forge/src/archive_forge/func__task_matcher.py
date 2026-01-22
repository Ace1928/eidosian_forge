import os
from taskflow import formatters
from taskflow.listeners import base
from taskflow import logging
from taskflow import states
from taskflow import task
from taskflow.types import failure
from taskflow.utils import misc
def _task_matcher(node):
    item = node.item
    return isinstance(item, task.Task) and item.name == task_name