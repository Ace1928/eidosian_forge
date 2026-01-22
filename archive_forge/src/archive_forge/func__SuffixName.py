import copy
import os
def _SuffixName(name, suffix):
    """Add a suffix to the end of a target.

  Arguments:
    name: name of the target (foo#target)
    suffix: the suffix to be added
  Returns:
    Target name with suffix added (foo_suffix#target)
  """
    parts = name.rsplit('#', 1)
    parts[0] = f'{parts[0]}_{suffix}'
    return '#'.join(parts)