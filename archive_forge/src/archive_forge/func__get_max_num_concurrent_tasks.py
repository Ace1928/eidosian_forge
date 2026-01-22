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
def _get_max_num_concurrent_tasks(spark_context: SparkContext) -> int:
    """Gets the current max number of concurrent tasks."""
    if spark_context._jsc.sc().version() >= '3.1':
        return spark_context._jsc.sc().maxNumConcurrentTasks(spark_context._jsc.sc().resourceProfileManager().resourceProfileFromId(0))
    return spark_context._jsc.sc().maxNumConcurrentTasks()