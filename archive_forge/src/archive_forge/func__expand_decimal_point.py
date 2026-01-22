import re
def _expand_decimal_point(self, m: str) -> str:
    """
        This method is used to expand '.' into spoken word ' point '.
        """
    return m.group(1).replace('.', ' point ')