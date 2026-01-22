import os
from taskflow import formatters
from taskflow.listeners import base
from taskflow import logging
from taskflow import states
from taskflow import task
from taskflow.types import failure
from taskflow.utils import misc
def _make_matcher(task_name):
    """Returns a function that matches a node with task item with same name."""

    def _task_matcher(node):
        item = node.item
        return isinstance(item, task.Task) and item.name == task_name
    return _task_matcher