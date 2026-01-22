import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from os.path import expanduser, join
@staticmethod
def _get_spark_task_context_or_none():
    try:
        from pyspark import TaskContext
        return TaskContext.get()
    except ImportError:
        return None