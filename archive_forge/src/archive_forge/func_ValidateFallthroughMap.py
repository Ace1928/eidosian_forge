import copy
from googlecloudsdk.calliope.concepts import deps as deps_lib
def ValidateFallthroughMap(fallthroughs_map):
    """Validates fallthrough map to ensure fallthrough map is not invalid.

  Fallthrough maps are only invalid if an inactive fallthrough comes before
  an active fallthrough. It could result in an active fallthrough that can
  never be reached.

  Args:
    fallthroughs_map: {str: [deps._FallthroughBase]}, A map of attribute
      names to fallthroughs we are validating

  Returns:
    (bool, str), bool for whether fallthrough map is valid and str for
      the error message
  """
    for attr, fallthroughs in fallthroughs_map.items():
        inactive_fallthrough = None
        for fallthrough in fallthroughs:
            if inactive_fallthrough and fallthrough.active:
                active_str = fallthrough.__class__.__name__
                inactive_str = inactive_fallthrough.__class__.__name__
                msg = f'Invalid Fallthrough Map: Fallthrough map at [{attr}] contains inactive fallthrough [{inactive_str}] before active fallthrough [{active_str}]. Fix the order so that active fallthrough [{active_str}] is reachable or remove active fallthrough [{active_str}].'
                return (False, msg)
            if not fallthrough.active:
                inactive_fallthrough = fallthrough
        else:
            return (True, None)