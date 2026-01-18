import logging
import psutil
import time
from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class ResourceUsage:
    """📊👀 Detailed tracking of resource consumption with explicit type annotations.

    Attributes:
        cpu_percent: The percentage of CPU utilization.
        memory_percent: The percentage of memory utilization.
        disk_percent: The percentage of disk utilization.
        resident_memory: The resident memory usage in bytes.
        virtual_memory: The virtual memory usage in bytes.
        timestamp: The timestamp of when the resource usage was recorded.
    """

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    resident_memory: int = 0
    virtual_memory: int = 0
    timestamp: float = field(default_factory=time.time)

    def __str__(self):
        """Returns a string representation of the resource usage."""
        return (
            f"ResourceUsage(cpu_percent={self.cpu_percent:.2f}%, "
            f"memory_percent={self.memory_percent:.2f}%, "
            f"disk_percent={self.disk_percent:.2f}%, "
            f"resident_memory={self.resident_memory} bytes, "
            f"virtual_memory={self.virtual_memory} bytes, "
            f"timestamp={self.timestamp})"
        )


def _get_resource_usage() -> ResourceUsage:
    """Collects and returns current resource usage statistics."""
    try:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")  # Assuming root partition for disk usage
        resource_usage = ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            resident_memory=psutil.Process().memory_info().rss,
            virtual_memory=memory.total,
        )
        return resource_usage
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error getting resource usage: {e}", exc_info=True)
        return ResourceUsage()
