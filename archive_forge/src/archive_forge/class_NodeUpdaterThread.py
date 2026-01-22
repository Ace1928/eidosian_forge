import logging
import os
import subprocess
import time
import traceback
from threading import Thread
import click
from ray._private.usage import usage_constants, usage_lib
from ray.autoscaler._private import subprocess_output_util as cmd_output_util
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.command_runner import (
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.log_timer import LogTimer
from ray.autoscaler.tags import (
class NodeUpdaterThread(NodeUpdater, Thread):

    def __init__(self, *args, **kwargs):
        Thread.__init__(self)
        NodeUpdater.__init__(self, *args, **kwargs)
        self.exitcode = -1