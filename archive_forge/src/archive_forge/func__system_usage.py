import logging
from ray._private.utils import get_num_cpus
def _system_usage():
    """
    Computes total CPU usage of the host in nanoseconds.

    Logic taken from here:
    https://github.com/moby/moby/blob/b42ac8d370a8ef8ec720dff0ca9dfb3530ac0a6a/daemon/stats/collector_unix.go#L31

    See also the /proc/stat entry here:
    https://man7.org/linux/man-pages/man5/proc.5.html
    """
    cpu_summary_str = open(PROC_STAT_PATH).read().split('\n')[0]
    parts = cpu_summary_str.split()
    assert parts[0] == 'cpu'
    usage_data = parts[1:8]
    total_clock_ticks = sum((int(entry) for entry in usage_data))
    usage_ns = total_clock_ticks * 10 ** 7
    return usage_ns