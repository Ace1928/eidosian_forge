import re
def _add_capacity(pool):
    """Add capacity formatting.

  Args:
    pool: the serializable storage pool
  Returns:
    nothing, it changes the input value.
  """
    provisioned_capacity_bytes = int(pool['poolProvisionedCapacityGb']) * GB
    provisioned_capacity_tb = provisioned_capacity_bytes / TB
    used_capacity_bytes = int(pool['status']['poolUsedCapacityBytes'])
    used_capacity_tb = used_capacity_bytes / TB
    formatted_capacity = '{:,.1f}/{:,.0f} ({:.1f}%)'.format(used_capacity_tb, provisioned_capacity_tb, 100 * (used_capacity_bytes / provisioned_capacity_bytes))
    pool['formattedCapacity'] = formatted_capacity