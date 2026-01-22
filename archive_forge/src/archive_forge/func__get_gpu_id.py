import inspect
import logging
import os
import sys
import uuid
from threading import Thread
from typing import Any, Callable, Dict, Optional, Set, Type
import pyspark
from pyspark import BarrierTaskContext, SparkContext, SparkFiles, TaskContext
from pyspark.sql.session import SparkSession
from xgboost import Booster, XGBModel, collective
from xgboost.tracker import RabitTracker
def _get_gpu_id(task_context: TaskContext) -> int:
    """Get the gpu id from the task resources"""
    if task_context is None:
        raise RuntimeError('_get_gpu_id should not be invoked from driver side.')
    resources = task_context.resources()
    if 'gpu' not in resources:
        raise RuntimeError("Couldn't get the gpu id, Please check the GPU resource configuration")
    return int(resources['gpu'].addresses[0].strip())