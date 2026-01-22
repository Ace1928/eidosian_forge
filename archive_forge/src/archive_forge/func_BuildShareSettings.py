from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis.arg_utils import ChoiceToEnumName
def BuildShareSettings(messages, args):
    """Build ShareSettings object from parameters."""
    if args.share_setting == 'projects' and (not args.share_with):
        msg = '[--share-setting=projects] must be specified with [--share-with]'
        raise exceptions.RequiredArgumentException('--share-with', msg)
    if (args.share_setting == 'organization' or args.share_setting == 'local') and args.share_with:
        msg = 'List of shared projects must be empty for {} share type'.format(args.share_setting)
        raise exceptions.InvalidArgumentException('--share-with', msg)
    if args.share_setting == 'projects':
        additional_properties = []
        for project in args.share_with:
            additional_properties.append(messages.ShareSettings.ProjectMapValue.AdditionalProperty(key=project, value=messages.ShareSettingsProjectConfig(projectId=project)))
        return messages.ShareSettings(shareType=messages.ShareSettings.ShareTypeValueValuesEnum.SPECIFIC_PROJECTS, projectMap=messages.ShareSettings.ProjectMapValue(additionalProperties=additional_properties))
    elif args.share_setting == 'organization':
        return messages.ShareSettings(shareType=messages.ShareSettings.ShareTypeValueValuesEnum.ORGANIZATION)
    return messages.ShareSettings(shareType=messages.ShareSettings.ShareTypeValueValuesEnum.LOCAL)