import glob
import logging
import os
import platform
import re
import subprocess
import sys
from argparse import ArgumentParser, RawTextHelpFormatter, REMAINDER
from os.path import expanduser
from typing import Dict, List
from torch.distributed.elastic.multiprocessing import start_processes, Std
class _CPUinfo:
    """Get CPU information, such as cores list and NUMA information."""

    def __init__(self, test_input=''):
        self.cpuinfo = []
        if platform.system() in ['Windows', 'Darwin']:
            raise RuntimeError(f'{platform.system()} is not supported!!!')
        elif platform.system() == 'Linux':
            if test_input == '':
                lscpu_cmd = ['lscpu', '--parse=CPU,Core,Socket,Node']
                lscpu_info = subprocess.check_output(lscpu_cmd, universal_newlines=True).split('\n')
            else:
                lscpu_info = test_input.split('\n')
            for line in lscpu_info:
                pattern = '^([\\d]+,[\\d]+,[\\d]+,[\\d]?)'
                regex_out = re.search(pattern, line)
                if regex_out:
                    self.cpuinfo.append(regex_out.group(1).strip().split(','))
            self.node_nums = int(max([line[3] for line in self.cpuinfo])) + 1
            self.node_physical_cores: List[List[int]] = []
            self.node_logical_cores: List[List[int]] = []
            self.physical_core_node_map = {}
            self.logical_core_node_map = {}
            for node_id in range(self.node_nums):
                cur_node_physical_core = []
                cur_node_logical_core = []
                for cpuinfo in self.cpuinfo:
                    nid = cpuinfo[3] if cpuinfo[3] != '' else '0'
                    if node_id == int(nid):
                        if int(cpuinfo[1]) not in cur_node_physical_core:
                            cur_node_physical_core.append(int(cpuinfo[1]))
                            self.physical_core_node_map[int(cpuinfo[1])] = int(node_id)
                        cur_node_logical_core.append(int(cpuinfo[0]))
                        self.logical_core_node_map[int(cpuinfo[0])] = int(node_id)
                self.node_physical_cores.append(cur_node_physical_core)
                self.node_logical_cores.append(cur_node_logical_core)

    def _physical_core_nums(self):
        return len(self.node_physical_cores) * len(self.node_physical_cores[0])

    def _logical_core_nums(self):
        return len(self.node_logical_cores) * len(self.node_logical_cores[0])

    def get_node_physical_cores(self, node_id):
        if node_id < 0 or node_id > self.node_nums - 1:
            raise ValueError(f'Invalid node id: {node_id}. Valid node ids: {list(range(len(self.node_physical_cores)))}')
        return self.node_physical_cores[node_id]

    def get_node_logical_cores(self, node_id):
        if node_id < 0 or node_id > self.node_nums - 1:
            raise ValueError(f'Invalid node id: {node_id}. Valid node ids: {list(range(len(self.node_physical_cores)))}')
        return self.node_logical_cores[node_id]

    def get_all_physical_cores(self):
        all_cores = []
        for cores in self.node_physical_cores:
            all_cores.extend(cores)
        return all_cores

    def get_all_logical_cores(self):
        all_cores = []
        for cores in self.node_logical_cores:
            all_cores.extend(cores)
        return all_cores

    def numa_aware_check(self, core_list):
        """
        Check whether all cores in core_list are in the same NUMA node.

        Cross NUMA will reduce performance.
        We strongly advice to not use cores on different nodes.
        """
        cores_numa_map = self.logical_core_node_map
        numa_ids = []
        for core in core_list:
            numa_id = cores_numa_map[core]
            if numa_id not in numa_ids:
                numa_ids.append(numa_id)
        if len(numa_ids) > 1:
            logger.warning('Numa Aware: cores:%s on different NUMA nodes:%s. To avoid this behavior, please use --ncores-per-instance knob to make sure number of cores is divisible by --ncores-per-instance. Alternatively, please use --skip-cross-node-cores knob.', str(core_list), str(numa_ids))
        if len(numa_ids) == 0:
            raise RuntimeError('invalid number of NUMA nodes; please make sure numa_ids >= 1')
        return numa_ids