from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from typing import List
from googlecloudsdk.api_lib.dataplex import util as dataplex_util
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
def ParseAspects(aspects_file: str) -> dataplex_message.GoogleCloudDataplexV1Entry.AspectsValue:
    """Parse aspects from a YAML or JSON file.

  Perform a basic validation that aspects are provided in a correct format.

  Args:
    aspects_file: The path to the YAML/JSON file containing aspects.

  Returns:
    A list of aspects parsed to a proto message (AspectsValue).
  """
    parser = arg_parsers.YAMLFileContents()
    raw_aspects = parser(aspects_file)
    if not isinstance(raw_aspects, dict):
        raise arg_parsers.ArgumentTypeError(f"Invalid aspects file: {aspects_file}. It must contain a map with a key in the format `ASPECT_TYPE@PATH` (or just `ASPECT_TYPE` if attached to the root path). Values in the map represent Aspect's content, which must conform to a template defined for a given `ASPECT_TYPE`.")
    aspects = []
    for aspect_key, aspect in raw_aspects.items():
        aspects.append(dataplex_message.GoogleCloudDataplexV1Entry.AspectsValue.AdditionalProperty(key=aspect_key, value=messages_util.DictToMessageWithErrorCheck(aspect, dataplex_message.GoogleCloudDataplexV1Aspect)))
    return dataplex_message.GoogleCloudDataplexV1Entry.AspectsValue(additionalProperties=aspects)