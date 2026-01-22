import enum
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def _validate_namespace_whitelist(namespace_whitelist):
    """Validates namespace whitelist argument."""
    if namespace_whitelist is None:
        return None
    if not isinstance(namespace_whitelist, list):
        raise TypeError(f'`namespace_whitelist` must be a list of strings. Got: {namespace_whitelist} with type {type(namespace_whitelist)}.')
    processed = []
    for namespace in namespace_whitelist:
        if not isinstance(namespace, str):
            raise ValueError(f'Whitelisted namespace must be a string. Got: {namespace} of type {type(namespace)}.')
        processed.append(compat.as_str(namespace))
    return processed