from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
import six
from gslib.cloud_api import NotFoundException
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import StorageUrlFromString
from gslib.utils.constants import NO_MAX
from gslib.utils.constants import S3_DELETE_MARKER_GUID
from gslib.utils.constants import UTF8
from gslib.utils.ls_helper import ENCRYPTED_FIELDS
from gslib.utils.ls_helper import LsHelper
from gslib.utils.ls_helper import PrintFullInfoAboutObject
from gslib.utils.ls_helper import UNENCRYPTED_FULL_LISTING_FIELDS
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.text_util import InsistAscii
from gslib.utils import text_util
from gslib.utils.translation_helper import AclTranslation
from gslib.utils.translation_helper import LabelTranslation
from gslib.utils.unit_util import MakeHumanReadable
def _PrintBucketInfo(self, bucket_blr, listing_style):
    """Print listing info for given bucket.

    Args:
      bucket_blr: BucketListingReference for the bucket being listed
      listing_style: ListingStyle enum describing type of output desired.

    Returns:
      Tuple (total objects, total bytes) in the bucket.
    """
    if listing_style == ListingStyle.SHORT or listing_style == ListingStyle.LONG:
        text_util.print_to_fd(bucket_blr)
        return
    bucket = bucket_blr.root_object
    location_constraint = bucket.location
    storage_class = bucket.storageClass
    fields = {'bucket': bucket_blr.url_string, 'storage_class': storage_class, 'location_constraint': location_constraint, 'acl': AclTranslation.JsonFromMessage(bucket.acl), 'default_acl': AclTranslation.JsonFromMessage(bucket.defaultObjectAcl), 'versioning': bucket.versioning and bucket.versioning.enabled, 'website_config': 'Present' if bucket.website else 'None', 'logging_config': 'Present' if bucket.logging else 'None', 'cors_config': 'Present' if bucket.cors else 'None', 'lifecycle_config': 'Present' if bucket.lifecycle else 'None', 'requester_pays': bucket.billing and bucket.billing.requesterPays}
    if bucket.retentionPolicy:
        fields['retention_policy'] = 'Present'
    if bucket.labels:
        fields['labels'] = LabelTranslation.JsonFromMessage(bucket.labels, pretty_print=True)
    else:
        fields['labels'] = 'None'
    if bucket.encryption and bucket.encryption.defaultKmsKeyName:
        fields['default_kms_key'] = bucket.encryption.defaultKmsKeyName
    else:
        fields['default_kms_key'] = 'None'
    fields['encryption_config'] = 'Present' if bucket.encryption else 'None'
    if bucket.autoclass and bucket.autoclass.enabled:
        fields['autoclass_enabled_date'] = bucket.autoclass.toggleTime.strftime('%a, %d %b %Y')
    if bucket.locationType:
        fields['location_type'] = bucket.locationType
    if bucket.customPlacementConfig:
        fields['custom_placement_locations'] = bucket.customPlacementConfig.dataLocations
    if bucket.metageneration:
        fields['metageneration'] = bucket.metageneration
    if bucket.timeCreated:
        fields['time_created'] = bucket.timeCreated.strftime('%a, %d %b %Y %H:%M:%S GMT')
    if bucket.updated:
        fields['updated'] = bucket.updated.strftime('%a, %d %b %Y %H:%M:%S GMT')
    if bucket.defaultEventBasedHold:
        fields['default_eventbased_hold'] = bucket.defaultEventBasedHold
    if bucket.iamConfiguration:
        if bucket.iamConfiguration.bucketPolicyOnly:
            enabled = bucket.iamConfiguration.bucketPolicyOnly.enabled
            fields['bucket_policy_only_enabled'] = enabled
        if bucket.iamConfiguration.publicAccessPrevention:
            fields['public_access_prevention'] = bucket.iamConfiguration.publicAccessPrevention
    if bucket.rpo:
        fields['rpo'] = bucket.rpo
    if bucket.satisfiesPZS:
        fields['satisfies_pzs'] = bucket.satisfiesPZS
    for key in fields:
        previous_value = fields[key]
        if not isinstance(previous_value, six.string_types) or '\n' not in previous_value:
            continue
        new_value = previous_value.replace('\n', '\n\t  ')
        if not new_value.startswith('\n'):
            new_value = '\n\t  ' + new_value
        fields[key] = new_value
    autoclass_line = ''
    location_type_line = ''
    custom_placement_locations_line = ''
    metageneration_line = ''
    time_created_line = ''
    time_updated_line = ''
    default_eventbased_hold_line = ''
    retention_policy_line = ''
    bucket_policy_only_enabled_line = ''
    public_access_prevention_line = ''
    rpo_line = ''
    satisifies_pzs_line = ''
    if 'autoclass_enabled_date' in fields:
        autoclass_line = '\tAutoclass:\t\t\tEnabled on {autoclass_enabled_date}\n'
    if 'location_type' in fields:
        location_type_line = '\tLocation type:\t\t\t{location_type}\n'
    if 'custom_placement_locations' in fields:
        custom_placement_locations_line = '\tPlacement locations:\t\t{custom_placement_locations}\n'
    if 'metageneration' in fields:
        metageneration_line = '\tMetageneration:\t\t\t{metageneration}\n'
    if 'time_created' in fields:
        time_created_line = '\tTime created:\t\t\t{time_created}\n'
    if 'updated' in fields:
        time_updated_line = '\tTime updated:\t\t\t{updated}\n'
    if 'default_eventbased_hold' in fields:
        default_eventbased_hold_line = '\tDefault Event-Based Hold:\t{default_eventbased_hold}\n'
    if 'retention_policy' in fields:
        retention_policy_line = '\tRetention Policy:\t\t{retention_policy}\n'
    if 'bucket_policy_only_enabled' in fields:
        bucket_policy_only_enabled_line = '\tBucket Policy Only enabled:\t{bucket_policy_only_enabled}\n'
    if 'public_access_prevention' in fields:
        public_access_prevention_line = '\tPublic access prevention:\t{public_access_prevention}\n'
    if 'rpo' in fields:
        rpo_line = '\tRPO:\t\t\t\t{rpo}\n'
    if 'satisfies_pzs' in fields:
        satisifies_pzs_line = '\tSatisfies PZS:\t\t\t{satisfies_pzs}\n'
    text_util.print_to_fd(('{bucket} :\n\tStorage class:\t\t\t{storage_class}\n' + location_type_line + '\tLocation constraint:\t\t{location_constraint}\n' + custom_placement_locations_line + '\tVersioning enabled:\t\t{versioning}\n\tLogging configuration:\t\t{logging_config}\n\tWebsite configuration:\t\t{website_config}\n\tCORS configuration: \t\t{cors_config}\n\tLifecycle configuration:\t{lifecycle_config}\n\tRequester Pays enabled:\t\t{requester_pays}\n' + retention_policy_line + default_eventbased_hold_line + '\tLabels:\t\t\t\t{labels}\n' + '\tDefault KMS key:\t\t{default_kms_key}\n' + time_created_line + time_updated_line + metageneration_line + bucket_policy_only_enabled_line + autoclass_line + public_access_prevention_line + rpo_line + satisifies_pzs_line + '\tACL:\t\t\t\t{acl}\n\tDefault ACL:\t\t\t{default_acl}').format(**fields))
    if bucket_blr.storage_url.scheme == 's3':
        text_util.print_to_fd('Note: this is an S3 bucket so configuration values may be blank. To retrieve bucket configuration values, use individual configuration commands such as gsutil acl get <bucket>.')