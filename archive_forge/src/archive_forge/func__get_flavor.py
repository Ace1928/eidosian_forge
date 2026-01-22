import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
def _get_flavor(cs, opts_str):
    flavor_name, opts_str = _strip_option(opts_str, 'flavor', True)
    flavor_id = _find_flavor(cs, flavor_name).id
    return (str(flavor_id), opts_str)