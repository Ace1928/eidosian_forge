import copy
from googlecloudsdk.calliope.concepts import deps as deps_lib
def UpdateWithValueFallthrough(base_fallthroughs_map, attribute_name, parsed_args):
    """Shortens fallthrough list to a single deps.ValueFallthrough.

  Used to replace the attribute_name entry in a fallthrough map to a
  single ValueFallthrough. For example:

    {'book': [deps.Fallthrough(lambda: 'foo')]}

  will update to something like...

    {'book': [deps.ValueFallthrough('foo')]}

  Args:
    base_fallthroughs_map: {str: [deps._FallthroughBase]}, A map of attribute
      names to fallthroughs we are updating
    attribute_name: str, entry in fallthrough map we are updating
    parsed_args: Namespace | None, used to derive the value for ValueFallthrough
  """
    if not parsed_args:
        return
    attribute_value, attribute_fallthrough = _GetFallthroughAndValue(attribute_name, base_fallthroughs_map, parsed_args)
    if attribute_fallthrough:
        _UpdateMapWithValueFallthrough(base_fallthroughs_map, attribute_value, attribute_name, attribute_fallthrough)