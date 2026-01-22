import re
def _format_pool_type(pool):
    """Format pool type.

  Args:
    pool: the serializable storage pool
  Returns:
    the formatted string
  """
    try:
        matched_type = STORAGE_POOL_TYPE_REGEX.search(pool['storagePoolType']).group(1).lower()
    except IndexError:
        return UNKNOWN_TYPE_PLACEHOLDER
    return STORAGE_POOL_TYPE_ABBREVIATIONS.get(matched_type, UNKNOWN_TYPE_PLACEHOLDER)