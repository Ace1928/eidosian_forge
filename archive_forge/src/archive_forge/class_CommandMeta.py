import abc
import logging
from cliff import command
from cliff import lister
from cliff import show
from osc_lib import exceptions
from osc_lib.i18n import _
class CommandMeta(abc.ABCMeta):

    def __new__(mcs, name, bases, cls_dict):
        if 'log' not in cls_dict:
            cls_dict['log'] = logging.getLogger(cls_dict['__module__'] + '.' + name)
        return super(CommandMeta, mcs).__new__(mcs, name, bases, cls_dict)