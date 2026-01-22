import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def _pretty_format_dict(data_dict):
    formatted_data = []
    for k in sorted(data_dict):
        formatted_data.append('%s : %s' % (k, data_dict[k]))
    return '\n'.join(formatted_data)