from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import os.path
import string
import uuid
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import daisy_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.images import flags
from googlecloudsdk.command_lib.compute.images import os_choices
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
import six
class ImportFromExternalCloudProviderStager(BaseImportStager):
    """Image import stager from an external cloud provider."""

    def Stage(self):
        import_args = []
        _AppendAwsArgs(self.args, import_args)
        _AppendTranslateWorkflowArg(self.args, import_args)
        import_args.extend(super(ImportFromExternalCloudProviderStager, self).Stage())
        return import_args

    def GetBucketLocation(self):
        if self.args.zone:
            return daisy_utils.GetRegionFromZone(self.args.zone)
        return super(ImportFromExternalCloudProviderStager, self).GetBucketLocation()