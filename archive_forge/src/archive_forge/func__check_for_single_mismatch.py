import re
@staticmethod
def _check_for_single_mismatch(data, regex):
    if regex is None:
        return False
    if not isinstance(data, str):
        return True
    if not regex.match(data):
        return True
    return False