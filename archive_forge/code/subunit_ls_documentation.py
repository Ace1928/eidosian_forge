import sys
from optparse import OptionParser
from testtools import CopyStreamResult, StreamResultRouter, StreamSummary
from subunit import ByteStreamToStreamResult
from subunit.filters import find_stream
from subunit.test_results import CatFiles, TestIdPrintingResult
List tests in a subunit stream.