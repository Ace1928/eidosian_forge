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
def use_cuda(device: Optional[str]) -> bool:
    """Whether xgboost is using CUDA workers."""
    return device in ('cuda', 'gpu')