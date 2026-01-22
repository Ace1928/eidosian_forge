import logging
import os
import time
from datetime import datetime
from shlex import quote, split
from threading import RLock
import kubernetes
from kubernetes.config.config_exception import ConfigException
import docker
from mlflow.entities import RunStatus
from mlflow.exceptions import ExecutionException
from mlflow.projects.submitted_run import SubmittedRun
def _load_kube_context(context=None):
    try:
        kubernetes.config.load_kube_config(context=context)
    except (OSError, ConfigException) as e:
        _logger.debug('Error loading kube context "%s": %s', context, e)
        _logger.info('No valid kube config found, using in-cluster configuration')
        kubernetes.config.load_incluster_config()