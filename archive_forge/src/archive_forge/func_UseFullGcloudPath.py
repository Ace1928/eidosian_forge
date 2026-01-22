from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.source import git
from googlecloudsdk.api_lib.source import sourcerepo
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.core import log
from googlecloudsdk.core.credentials import store as c_store
def UseFullGcloudPath(self, args):
    """Use value of --use-full-gcloud-path argument in beta and alpha."""
    return args.use_full_gcloud_path