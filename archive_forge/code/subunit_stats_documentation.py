import sys
from testtools import StreamToExtendedDecorator
from subunit import TestResultStats
from subunit.filters import run_filter_script
Filter a subunit stream to get aggregate statistics.