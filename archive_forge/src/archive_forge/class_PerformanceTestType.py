import enum
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
class PerformanceTestType(enum.Enum):
    """Enum class for specifying performance test type for diagnostic tests."""
    DOWNLOAD_THROUGHPUT = 'DOWNLOAD_THROUGHPUT'
    UPLOAD_THROUGHPUT = 'UPLOAD_THROUGHPUT'
    LATENCY = 'LATENCY'