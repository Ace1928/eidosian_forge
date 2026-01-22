from openstack import exceptions
from openstack import resource
def _dict_filter(f, d):
    """Dict param based filtering"""
    if not d:
        return False
    for key in f.keys():
        if isinstance(f[key], dict):
            if not _dict_filter(f[key], d.get(key, None)):
                return False
        elif d.get(key, None) != f[key]:
            return False
    return True