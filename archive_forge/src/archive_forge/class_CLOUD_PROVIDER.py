import json
import urllib3
from sentry_sdk.integrations import Integration
from sentry_sdk.api import set_context
from sentry_sdk.utils import logger
from sentry_sdk._types import TYPE_CHECKING
class CLOUD_PROVIDER:
    """
    Name of the cloud provider.
    see https://opentelemetry.io/docs/reference/specification/resource/semantic_conventions/cloud/
    """
    ALIBABA = 'alibaba_cloud'
    AWS = 'aws'
    AZURE = 'azure'
    GCP = 'gcp'
    IBM = 'ibm_cloud'
    TENCENT = 'tencent_cloud'