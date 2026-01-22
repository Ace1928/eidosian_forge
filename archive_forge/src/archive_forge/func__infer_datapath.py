import argparse
import importlib
import os
import sys as _sys
import datetime
import parlai
import parlai.utils.logging as logging
from parlai.core.build_data import modelzoo_path
from parlai.core.loader import (
from parlai.tasks.tasks import ids_to_tasks
from parlai.core.opt import Opt
from typing import List, Optional
def _infer_datapath(self, opt):
    """
        Set the value for opt['datapath'] and opt['download_path'].

        Sets the value for opt['datapath'] and opt['download_path'], correctly
        respecting environmental variables and the default.
        """
    if opt.get('download_path'):
        os.environ['PARLAI_DOWNPATH'] = opt['download_path']
    elif os.environ.get('PARLAI_DOWNPATH') is None:
        os.environ['PARLAI_DOWNPATH'] = os.path.join(self.parlai_home, 'downloads')
    if opt.get('datapath'):
        os.environ['PARLAI_DATAPATH'] = opt['datapath']
    elif os.environ.get('PARLAI_DATAPATH') is None:
        os.environ['PARLAI_DATAPATH'] = os.path.join(self.parlai_home, 'data')
    opt['download_path'] = os.environ['PARLAI_DOWNPATH']
    opt['datapath'] = os.environ['PARLAI_DATAPATH']
    return opt