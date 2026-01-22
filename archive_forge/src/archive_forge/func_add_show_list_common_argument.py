import abc
import argparse
import functools
import logging
from cliff import command
from cliff import lister
from cliff import show
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
def add_show_list_common_argument(parser):
    parser.add_argument('-D', '--show-details', help=_('Show detailed information.'), action='store_true', default=False)
    parser.add_argument('--show_details', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--fields', help=argparse.SUPPRESS, action='append', default=[])
    parser.add_argument('-F', '--field', dest='fields', metavar='FIELD', help=_('Specify the field(s) to be returned by server. You can repeat this option.'), action='append', default=[])