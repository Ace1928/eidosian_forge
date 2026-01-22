from apitools.base.py import list_pager
from googlecloudsdk.api_lib.quotas import message_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import common_args
def UpdateQuotaPreference(args):
    """Updates the parameters of a single QuotaPreference.

  Args:
    args: argparse.Namespace, The arguments that this command was invoked with.

  Returns:
    The updated QuotaPreference.
  """
    consumer = message_util.CreateConsumer(args.project, args.folder, args.organization)
    client = _GetClientInstance()
    messages = client.MESSAGES_MODULE
    preference_name = _GetPreferenceName(_CONSUMER_LOCATION_RESOURCE % consumer, args.PREFERENCE_ID)
    quota_preference = messages.QuotaPreference(name=preference_name, dimensions=_GetDimensions(messages, args.dimensions), quotaConfig=messages.QuotaConfig(preferredValue=int(args.preferred_value)), service=args.service, quotaId=args.quota_id, justification=_GetJustification(args.email, args.justification), contactEmail=args.email)
    if args.project:
        request = messages.CloudquotasProjectsLocationsQuotaPreferencesPatchRequest(name=preference_name, quotaPreference=quota_preference, allowMissing=args.allow_missing, validateOnly=args.validate_only, ignoreSafetyChecks=_GetIgnoreSafetyChecks(args, messages.CloudquotasProjectsLocationsQuotaPreferencesPatchRequest))
        return client.projects_locations_quotaPreferences.Patch(request)
    if args.folder:
        request = messages.CloudquotasFoldersLocationsQuotaPreferencesPatchRequest(name=preference_name, quotaPreference=quota_preference, allowMissing=args.allow_missing, validateOnly=args.validate_only, ignoreSafetyChecks=_GetIgnoreSafetyChecks(args, messages.CloudquotasFoldersLocationsQuotaPreferencesPatchRequest))
        return client.folders_locations_quotaPreferences.Patch(request)
    if args.organization:
        request = messages.CloudquotasOrganizationsLocationsQuotaPreferencesPatchRequest(name=preference_name, quotaPreference=quota_preference, allowMissing=args.allow_missing, validateOnly=args.validate_only, ignoreSafetyChecks=_GetIgnoreSafetyChecks(args, messages.CloudquotasOrganizationsLocationsQuotaPreferencesPatchRequest))
        return client.organizations_locations_quotaPreferences.Patch(request)