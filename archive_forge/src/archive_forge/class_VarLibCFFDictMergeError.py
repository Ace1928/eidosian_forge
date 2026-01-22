import textwrap
class VarLibCFFDictMergeError(VarLibCFFMergeError):
    """Raised when a CFF PrivateDict cannot be merged."""

    def __init__(self, key, value, values):
        error_msg = f"For the Private Dict key '{key}', the default font value list:\n\t{value}\nhad a different number of values than a region font:"
        for region_value in values:
            error_msg += f'\n\t{region_value}'
        self.args = (error_msg,)