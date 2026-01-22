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
def add_sorting_argument(parser):
    parser.add_argument('--sort-key', dest='sort_key', metavar='FIELD', action='append', help=_('Sorts the list by the specified fields in the specified directions. You can repeat this option, but you must specify an equal number of sort_dir and sort_key values. Extra sort_dir options are ignored. Missing sort_dir options use the default asc value.'), default=[])
    parser.add_argument('--sort-dir', dest='sort_dir', metavar='{asc,desc}', help=_('Sorts the list in the specified direction. You can repeat this option.'), action='append', default=[], choices=['asc', 'desc'])