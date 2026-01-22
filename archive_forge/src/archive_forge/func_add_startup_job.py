import os
import pwd
import re
import subprocess
import time
import xml.dom.minidom
import random
from ... import logging
from ...interfaces.base import CommandLine
from .base import SGELikeBatchManagerBase, logger
def add_startup_job(self, taskid, qsub_command_line):
    """
        :param taskid: The job id
        :param qsub_command_line: When initializing, re-use the job_queue_name
        :return: NONE
        """
    taskid = int(taskid)
    self._task_dictionary[taskid] = QJobInfo(taskid, 'initializing', time.time(), 'noQueue', 1, qsub_command_line)