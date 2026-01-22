from tensorflow.python.util.tf_export import tf_export
from tensorflow.python import pywrap_tfe
@staticmethod
def _string_to_components(spec=None):
    """Stateless portion of device spec string parsing.

    Args:
      spec: An optional string specifying a device specification.

    Returns:
      The parsed components of `spec`. Note that the result of this function
      must go through attribute setters of DeviceSpec, and should therefore NOT
      be used directly.
    """
    cached_result = _STRING_TO_COMPONENTS_CACHE.get(spec)
    if cached_result is not None:
        return cached_result
    raw_spec = spec
    job, replica, task, device_type, device_index = (None, None, None, None, None)
    spec = spec or ''
    splits = [x.split(':') for x in spec.split('/')]
    valid_device_types = DeviceSpecV2._get_valid_device_types()
    for y in splits:
        ly = len(y)
        if y:
            if ly == 2 and y[0] == 'job':
                job = y[1]
            elif ly == 2 and y[0] == 'replica':
                replica = y[1]
            elif ly == 2 and y[0] == 'task':
                task = y[1]
            elif (ly == 1 or ly == 2) and y[0].upper() in valid_device_types:
                if device_type is not None:
                    raise ValueError(f'Multiple device types are not allowed while parsing the device spec: {spec}.')
                device_type = y[0].upper()
                if ly == 2 and y[1] != '*':
                    device_index = int(y[1])
            elif ly == 3 and y[0] == 'device':
                if device_type is not None:
                    raise ValueError(f'Multiple device types are not allowed while parsing the device spec: {spec}.')
                device_type = y[1]
                if y[2] != '*':
                    device_index = int(y[2])
            elif ly and y[0] != '':
                raise ValueError(f"Unknown attribute '{y[0]}' is encountered while parsing the device spec: '{spec}'.")
    output = (job, replica, task, device_type, device_index)
    _STRING_TO_COMPONENTS_CACHE[raw_spec] = output
    return output