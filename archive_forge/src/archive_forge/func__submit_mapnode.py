import sys
from copy import deepcopy
from glob import glob
import os
import shutil
from time import sleep, time
from traceback import format_exception
import numpy as np
from ... import logging
from ...utils.misc import str2bool
from ..engine.utils import topological_sort, load_resultfile
from ..engine import MapNode
from .tools import report_crash, report_nodes_not_run, create_pyscript
def _submit_mapnode(self, jobid):
    import scipy.sparse as ssp
    if jobid in self.mapnodes:
        return True
    self.mapnodes.append(jobid)
    mapnodesubids = self.procs[jobid].get_subnodes()
    numnodes = len(mapnodesubids)
    logger.debug('Adding %d jobs for mapnode %s', numnodes, self.procs[jobid])
    for i in range(numnodes):
        self.mapnodesubids[self.depidx.shape[0] + i] = jobid
    self.procs.extend(mapnodesubids)
    self.depidx = ssp.vstack((self.depidx, ssp.lil_matrix(np.zeros((numnodes, self.depidx.shape[1])))), 'lil')
    self.depidx = ssp.hstack((self.depidx, ssp.lil_matrix(np.zeros((self.depidx.shape[0], numnodes)))), 'lil')
    self.depidx[-numnodes:, jobid] = 1
    self.proc_done = np.concatenate((self.proc_done, np.zeros(numnodes, dtype=bool)))
    self.proc_pending = np.concatenate((self.proc_pending, np.zeros(numnodes, dtype=bool)))
    return False