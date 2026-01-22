import logging
from osc_lib.command import command
from osc_lib import utils
from monascaclient import version
class MigratingCommandMeta(command.CommandMeta):
    """Overwrite module name based on osc_lib.CommandMeta requirements."""

    def __new__(mcs, name, bases, cls_dict):
        cls_dict['__module__'] = 'monascaclient.v2_0.shell'
        return super(MigratingCommandMeta, mcs).__new__(mcs, name, bases, cls_dict)