import copy
from googlecloudsdk.calliope.concepts import deps as deps_lib
def AddAnchorFallthroughs(base_fallthroughs_map, attributes, anchor, collection_info, anchor_fallthroughs):
    """Adds fully specified fallthroughs to fallthrough map.

  Iterates through each attribute and prepends a fully specified fallthrough.
  This allows resource attributes to resolve to the fully specified anchor
  value first. For example:

    {'book': [deps.ValueFallthrough('foo')]}

  will udpate to something like...

    {
        'book': [
            deps.FullySpecifiedAnchorFallthrough(anchor_fallthroughs),
            deps.ValueFallthrough('foo')
        ]
    }

  Args:
    base_fallthroughs_map: {str: [deps._FallthroughBase]}, A map of attribute
      names to fallthroughs
    attributes: list[concepts.Attribute], list of attributes associated
      with the resource
    anchor: concepts.Attribute, attribute that the other attributes should
      resolve to if fully specified
    collection_info: the info of the collection to parse the anchor as
    anchor_fallthroughs: list[deps._FallthroughBase], fallthroughs used to
      resolve the anchor value
  """
    for attribute in attributes:
        current_fallthroughs = base_fallthroughs_map.get(attribute.name, [])
        anchor_based_fallthrough = deps_lib.FullySpecifiedAnchorFallthrough(anchor_fallthroughs, collection_info, attribute.param_name)
        if attribute != anchor:
            filtered_fallthroughs = [f for f in current_fallthroughs if f != anchor_based_fallthrough]
            fallthroughs = [anchor_based_fallthrough] + filtered_fallthroughs
        else:
            fallthroughs = current_fallthroughs
        base_fallthroughs_map[attribute.name] = fallthroughs