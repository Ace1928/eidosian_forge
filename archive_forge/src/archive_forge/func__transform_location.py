import re
from googlecloudsdk.api_lib.storage import insights_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import flags
def _transform_location(dataset_config):
    matched_result = re.search(LOCATION_REGEX_PATTERN, dataset_config['name'])
    if matched_result and matched_result.group(1) is not None:
        return matched_result.group(1)
    else:
        return 'N/A-Misformated Value'