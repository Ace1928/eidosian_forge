import copy
from googlecloudsdk.calliope.concepts import deps as deps_lib
def PluralizeFallthroughs(base_fallthroughs_map, attribute_name):
    """Updates fallthrough map entry to make fallthroughs plural.

  For example:

    {'book': [deps.ArgFallthrough('--foo')]}

  will update to something like...

    {'book': [deps.ArgFallthrough('--foo'), plural=True]}

  Args:
    base_fallthroughs_map: {str: [deps.Fallthrough]}, A map of attribute
      names to fallthroughs we are updating
    attribute_name: str, entry in fallthrough map we are updating
  """
    given_fallthroughs = base_fallthroughs_map.get(attribute_name, [])
    base_fallthroughs_map[attribute_name] = [_PluralizeFallthrough(fallthrough) for fallthrough in given_fallthroughs]