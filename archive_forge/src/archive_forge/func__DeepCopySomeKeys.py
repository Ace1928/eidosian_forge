import copy
import os
def _DeepCopySomeKeys(in_dict, keys):
    """Performs a partial deep-copy on |in_dict|, only copying the keys in |keys|.

  Arguments:
    in_dict: The dictionary to copy.
    keys: The keys to be copied. If a key is in this list and doesn't exist in
        |in_dict| this is not an error.
  Returns:
    The partially deep-copied dictionary.
  """
    d = {}
    for key in keys:
        if key not in in_dict:
            continue
        d[key] = copy.deepcopy(in_dict[key])
    return d