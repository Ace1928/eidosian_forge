import abc
import os
import re
from googlecloudsdk.core.util import files
class SystemInfoProvider(abc.ABC):
    """Base system information provider.

  This class contains OS agnostic implemenations. Child classes may implement
  methods which are OS dependent.
  """

    def get_cpu_count(self) -> int:
        """Returns the number of logical CPUs in the system.

    Logical CPU is the number of threads that the OS can schedule work on.
    Includes physical cores and hyper-threaded cores.
    """
        return os.cpu_count()

    @abc.abstractmethod
    def get_cpu_load_avg(self) -> float:
        """Returns the average CPU load during last 1-minute."""
        pass

    @abc.abstractmethod
    def get_memory_stats(self) -> (int, int):
        """Fetches the physical memory stats for the system in bytes.

    Returns:
      A tuple containing total memory and free memory in the system
      respectively.
    """
        pass