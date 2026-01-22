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
@six.add_metaclass(abc.ABCMeta)
class BaseImportStager(object):
    """Base class for image import stager.

  An abstract class which is responsible for preparing import parameters, such
  as Daisy parameters and workflow, as well as creating Daisy scratch bucket in
  the appropriate location.
  """

    def __init__(self, storage_client, compute_holder, args):
        self.storage_client = storage_client
        self.compute_holder = compute_holder
        self.args = args
        self.daisy_bucket = self.GetAndCreateDaisyBucket()

    def Stage(self):
        """Prepares for import args.

    It supports running new import wrapper (gce_vm_image_import).

    Returns:
      import_args - array of strings, import args.
    """
        import_args = []
        messages = self.compute_holder.client.messages
        daisy_utils.AppendArg(import_args, 'zone', properties.VALUES.compute.zone.Get())
        if self.args.storage_location:
            daisy_utils.AppendArg(import_args, 'storage_location', self.args.storage_location)
        daisy_utils.AppendArg(import_args, 'scratch_bucket_gcs_path', 'gs://{0}/'.format(self.daisy_bucket))
        daisy_utils.AppendArg(import_args, 'timeout', '{}s'.format(daisy_utils.GetDaisyTimeout(self.args)))
        daisy_utils.AppendArg(import_args, 'client_id', 'gcloud')
        daisy_utils.AppendArg(import_args, 'image_name', self.args.image_name)
        daisy_utils.AppendBoolArg(import_args, 'no_guest_environment', not self.args.guest_environment)
        daisy_utils.AppendNetworkAndSubnetArgs(self.args, import_args)
        daisy_utils.AppendArg(import_args, 'description', self.args.description)
        daisy_utils.AppendArg(import_args, 'family', self.args.family)
        if 'byol' in self.args:
            daisy_utils.AppendBoolArg(import_args, 'byol', self.args.byol)
        guest_os_features = getattr(self.args, 'guest_os_features', None) or ()
        uefi_compatible = messages.GuestOsFeature.TypeValueValuesEnum.UEFI_COMPATIBLE.name in guest_os_features
        if uefi_compatible:
            daisy_utils.AppendBoolArg(import_args, 'uefi_compatible', True)
        if 'sysprep_windows' in self.args:
            daisy_utils.AppendBoolArg(import_args, 'sysprep_windows', self.args.sysprep_windows)
        if 'no_address' in self.args:
            daisy_utils.AppendBoolArg(import_args, 'no_external_ip', self.args.no_address)
        if 'compute_service_account' in self.args:
            daisy_utils.AppendArg(import_args, 'compute_service_account', self.args.compute_service_account)
        return import_args

    def GetAndCreateDaisyBucket(self):
        return daisy_utils.CreateDaisyBucketInProject(self.GetBucketLocation(), self.storage_client, enable_uniform_level_access=True)

    def GetBucketLocation(self):
        if self.args.storage_location:
            return self.args.storage_location
        return None