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
def add_pagination_argument(parser):
    parser.add_argument('-P', '--page-size', dest='page_size', metavar='SIZE', type=int, help=_('Specify retrieve unit of each request, then split one request to several requests.'), default=None)