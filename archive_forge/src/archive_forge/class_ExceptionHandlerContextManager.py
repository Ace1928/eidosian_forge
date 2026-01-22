from collections import defaultdict
from functools import lru_cache
import boto3
from boto3.exceptions import ResourceNotExistsError
from boto3.resources.base import ServiceResource
from botocore.client import BaseClient
from botocore.config import Config
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.constants import BOTO_MAX_RETRIES
class ExceptionHandlerContextManager:

    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        import botocore
        if type is botocore.exceptions.ClientError:
            handle_boto_error(value, msg, *args, **kwargs)