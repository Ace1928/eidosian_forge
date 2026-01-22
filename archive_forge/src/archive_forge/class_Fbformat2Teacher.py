import copy
import os
from parlai.core.teachers import (
class Fbformat2Teacher(FbDialogTeacher):
    """
    This task simply loads the specified file: useful for quick tests without setting up
    a new task.

    Used to set up a second task.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('FromFile Task Arguments')
        agent.add_argument('-dp', '--fromfile-datapath2', type=str, help='Data file')

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        if not opt.get('fromfile_datapath2'):
            raise RuntimeError('-dp', 'fromfile_datapath2 not specified')
        opt['datafile'] = opt['fromfile_datapath2']
        super().__init__(opt, shared)