from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import config
from googlecloudsdk.api_lib.cloudbuild import logs as cb_logs
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags
from googlecloudsdk.command_lib.builds import staging_bucket_util
from googlecloudsdk.command_lib.cloudbuild import execution
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
def _SetWorkerPoolConfig(build_config, messages, arg_disk_size, arg_memory, arg_vcpu_count):
    """Set the worker pool config."""
    if arg_disk_size is not None and cloudbuild_util.WorkerPoolIsSpecified(build_config) or arg_memory is not None or arg_vcpu_count is not None:
        if not build_config.options:
            build_config.options = messages.BuildOptions()
        if not build_config.options.pool:
            build_config.options.pool = messages.PoolOption()
        if not build_config.options.pool.workerConfig:
            build_config.options.pool.workerConfig = messages.GoogleDevtoolsCloudbuildV1BuildOptionsPoolOptionWorkerConfig()
        if arg_disk_size is not None:
            disk_size = compute_utils.BytesToGb(arg_disk_size)
            build_config.options.pool.workerConfig.diskSizeGb = disk_size
        if arg_memory is not None:
            memory = cloudbuild_util.BytesToGb(arg_memory)
            build_config.options.pool.workerConfig.memoryGb = memory
        if arg_vcpu_count is not None:
            build_config.options.pool.workerConfig.vcpuCount = arg_vcpu_count
    return build_config