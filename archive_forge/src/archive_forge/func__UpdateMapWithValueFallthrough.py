import copy
from googlecloudsdk.calliope.concepts import deps as deps_lib
def _UpdateMapWithValueFallthrough(base_fallthroughs_map, value, attribute_name, attribute_fallthrough):
    value_fallthrough = deps_lib.ValueFallthrough(value, attribute_fallthrough.hint, active=attribute_fallthrough.active)
    base_fallthroughs_map[attribute_name] = [value_fallthrough]