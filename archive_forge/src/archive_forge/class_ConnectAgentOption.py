from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import resources as cloud_resources
import six
class ConnectAgentOption(object):
    """Option for generating connect agent manifest."""

    def __init__(self, name, proxy, namespace, is_upgrade, version, registry, image_pull_secret_content, membership_ref):
        self.name = name
        self.proxy = proxy
        self.namespace = namespace
        self.is_upgrade = is_upgrade
        self.version = version
        self.registry = registry
        self.image_pull_secret_content = image_pull_secret_content
        self.membership_ref = membership_ref