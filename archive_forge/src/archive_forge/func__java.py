import platform as pf
from typing import Any, Iterable, Optional
from .metrics_core import GaugeMetricFamily, Metric
from .registry import Collector, CollectorRegistry, REGISTRY
def _java(self):
    java_version, _, vminfo, osinfo = self._platform.java_ver()
    vm_name, vm_release, vm_vendor = vminfo
    return {'jvm_version': java_version, 'jvm_release': vm_release, 'jvm_vendor': vm_vendor, 'jvm_name': vm_name}