import argparse
from cliff import command
from cliff import lister
from cliff import show
from oslo_serialization import jsonutils
from oslo_utils import strutils
from oslo_utils import uuidutils
from aodhclient import exceptions
from aodhclient.i18n import _
from aodhclient import utils
def validate_time_constraint(self, values_to_convert):
    """Converts 'a=1;b=2' to {a:1,b:2}."""
    try:
        return dict(((item.strip(' "\'') for item in kv.split('=', 1)) for kv in values_to_convert.split(';')))
    except ValueError:
        msg = 'must be a list of key1=value1;key2=value2;... not %s' % values_to_convert
        raise argparse.ArgumentTypeError(msg)