from .lib import TestBase, FileCreator
from smmap.util import (
import os
import sys
def test_region_list(self):
    with FileCreator(100, 'sample_file') as fc:
        fd = os.open(fc.path, os.O_RDONLY)
        try:
            for item in (fc.path, fd):
                ml = MapRegionList(item)
                assert len(ml) == 0
                assert ml.path_or_fd() == item
                assert ml.file_size() == fc.size
        finally:
            os.close(fd)