from apitools.base.py import list_pager
from googlecloudsdk.api_lib.quotas import message_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import common_args
def CreateQuotaPreference(args):
    """Creates a new QuotaPreference that declares the desired value for a quota.

  Args:
    args: argparse.Namespace, The arguments that this command was invoked with.

  Returns:
    The created QuotaPreference
  """
    consumer = message_util.CreateConsumer(args.project, args.folder, args.organization)
    client = _GetClientInstance()
    messages = client.MESSAGES_MODULE
    parent = _CONSUMER_LOCATION_RESOURCE % consumer
    quota_preference = messages.QuotaPreference(name=_GetPreferenceName(parent, args.preference_id), dimensions=_GetDimensions(messages, args.dimensions), quotaConfig=messages.QuotaConfig(preferredValue=int(args.preferred_value)), service=args.service, quotaId=args.quota_id, justification=_GetJustification(args.email, args.justification), contactEmail=args.email)
    if args.project:
        request = messages.CloudquotasProjectsLocationsQuotaPreferencesCreateRequest(parent=parent, quotaPreferenceId=args.preference_id, quotaPreference=quota_preference, ignoreSafetyChecks=_GetIgnoreSafetyChecks(args, messages.CloudquotasProjectsLocationsQuotaPreferencesCreateRequest))
        return client.projects_locations_quotaPreferences.Create(request)
    if args.folder:
        request = messages.CloudquotasFoldersLocationsQuotaPreferencesCreateRequest(parent=parent, quotaPreferenceId=args.preference_id, quotaPreference=quota_preference, ignoreSafetyChecks=_GetIgnoreSafetyChecks(args, messages.CloudquotasFoldersLocationsQuotaPreferencesCreateRequest))
        return client.folders_locations_quotaPreferences.Create(request)
    if args.organization:
        request = messages.CloudquotasOrganizationsLocationsQuotaPreferencesCreateRequest(parent=parent, quotaPreferenceId=args.preference_id, quotaPreference=quota_preference, ignoreSafetyChecks=_GetIgnoreSafetyChecks(args, messages.CloudquotasOrganizationsLocationsQuotaPreferencesCreateRequest))
        return client.organizations_locations_quotaPreferences.Create(request)