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
class ImportFromImageStager(BaseImportStager):
    """Image import stager from an existing image."""

    def Stage(self):
        _CheckForExistingImage(self.args.source_image, self.compute_holder, arg_name='source-image', expect_to_exist=True)
        import_args = []
        daisy_utils.AppendArg(import_args, 'source_image', self.args.source_image)
        if self.args.data_disk:
            daisy_utils.AppendBoolArg(import_args, 'data_disk', self.args.data_disk)
        else:
            _AppendTranslateWorkflowArg(self.args, import_args)
        import_args.extend(super(ImportFromImageStager, self).Stage())
        return import_args

    def _GetSourceImage(self):
        ref = resources.REGISTRY.Parse(self.args.source_image, collection='compute.images', params={'project': properties.VALUES.core.project.GetOrFail})
        source_name = ref.RelativeName()[len(ref.Parent().RelativeName() + '/'):]
        return source_name

    def GetBucketLocation(self):
        if self.args.zone:
            return daisy_utils.GetRegionFromZone(self.args.zone)
        return super(ImportFromImageStager, self).GetBucketLocation()