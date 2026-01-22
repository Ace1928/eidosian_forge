from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import crawlers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions
def SetUpdateMask(ref, args, request):
    """Python hook that computes the update mask for a patch request.

  Args:
    ref: The crawler resource reference.
    args: The parsed args namespace.
    request: The update crawler request.
  Returns:
    Request with update mask set appropriately.
  Raises:
    NoFieldsSpecifiedError: If no fields were provided for updating.
  """
    del ref
    update_mask = []
    if args.IsSpecified('description'):
        update_mask.append('description')
    if args.IsSpecified('display_name'):
        update_mask.append('displayName')
    if _IsChangeBundleSpecsSpecified(args):
        update_mask.append('config.bundleSpecs')
    if args.crawl_scope == 'project':
        update_mask.append('config.projectScope')
    elif args.crawl_scope == 'organization':
        update_mask.append('config.organizationScope')
    elif _IsChangeBucketsSpecified(args):
        update_mask.append('config.bucketScope')
    if args.run_option == 'manual':
        update_mask.append('config.adHocRun')
    elif args.run_option == 'scheduled' or args.IsSpecified('run_schedule'):
        update_mask.append('config.scheduledRun')
    if not update_mask:
        raise NoFieldsSpecifiedError('Must specify at least one parameter to update.')
    request.updateMask = ','.join(update_mask)
    return request