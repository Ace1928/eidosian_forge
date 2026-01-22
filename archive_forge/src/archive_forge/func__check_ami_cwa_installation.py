import copy
import logging
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List
import botocore
from boto3.resources.base import ServiceResource
import ray
import ray._private.ray_constants as ray_constants
from ray.autoscaler._private.aws.cloudwatch.cloudwatch_helper import (
from ray.autoscaler._private.aws.config import bootstrap_aws
from ray.autoscaler._private.aws.utils import (
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.constants import BOTO_CREATE_MAX_RETRIES, BOTO_MAX_RETRIES
from ray.autoscaler._private.log_timer import LogTimer
from ray.autoscaler.node_launch_exception import NodeLaunchException
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def _check_ami_cwa_installation(self, config):
    response = self.ec2.meta.client.describe_images(ImageIds=[config['ImageId']])
    cwa_installed = False
    images = response.get('Images')
    if images:
        assert len(images) == 1, f'Expected to find only 1 AMI with the given ID, but found {len(images)}.'
        image_name = images[0].get('Name', '')
        if CLOUDWATCH_AGENT_INSTALLED_AMI_TAG in image_name:
            cwa_installed = True
    return cwa_installed