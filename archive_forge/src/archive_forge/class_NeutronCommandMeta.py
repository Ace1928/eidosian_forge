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
class NeutronCommandMeta(abc.ABCMeta):

    def __new__(cls, name, bases, cls_dict):
        if 'log' not in cls_dict:
            cls_dict['log'] = logging.getLogger(cls_dict['__module__'] + '.' + name)
        return super(NeutronCommandMeta, cls).__new__(cls, name, bases, cls_dict)