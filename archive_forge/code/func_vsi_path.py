import sys
import re
from urllib.parse import urlparse
def vsi_path(path, vsi=None, archive=None):
    if vsi:
        prefix = '/'.join((f'vsi{SCHEMES[p]}' for p in vsi.split('+')))
        if archive:
            result = f'/{prefix}/{archive}{path}'
        else:
            result = f'/{prefix}/{path}'
    else:
        result = path
    return result