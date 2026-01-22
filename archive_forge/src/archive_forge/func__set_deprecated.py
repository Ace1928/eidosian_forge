import warnings
def _set_deprecated(self, value, *, new_key, deprecated_key, warning_message):
    """Set key in dictionary to be deprecated with its warning message."""
    self.__dict__['_deprecated_key_to_warnings'][deprecated_key] = warning_message
    self[new_key] = self[deprecated_key] = value