import logging
from ray._private.utils import get_num_cpus
def _host_num_cpus():
    """Number of physical CPUs, obtained by parsing /proc/stat."""
    global host_num_cpus
    if host_num_cpus is None:
        proc_stat_lines = open(PROC_STAT_PATH).read().split('\n')
        split_proc_stat_lines = [line.split() for line in proc_stat_lines]
        cpu_lines = [split_line for split_line in split_proc_stat_lines if len(split_line) > 0 and 'cpu' in split_line[0]]
        host_num_cpus = len(cpu_lines) - 1
    return host_num_cpus