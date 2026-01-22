import asyncio
import logging
import ray.dashboard.consts as dashboard_consts
from ray.dashboard.utils import (
class GlobalSignals:
    node_info_fetched = Signal(dashboard_consts.SIGNAL_NODE_INFO_FETCHED)
    node_summary_fetched = Signal(dashboard_consts.SIGNAL_NODE_SUMMARY_FETCHED)
    job_info_fetched = Signal(dashboard_consts.SIGNAL_JOB_INFO_FETCHED)
    worker_info_fetched = Signal(dashboard_consts.SIGNAL_WORKER_INFO_FETCHED)