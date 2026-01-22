import os
import shutil
import tempfile
import pyomo.common.envvar as envvar
from pyomo.common.fileutils import this_file_dir, find_dir
from pyomo.common.download import FileDownloader
class MCPPBuilder(object):

    def __call__(self, parallel):
        return build_mcpp()

    def skip(self):
        return FileDownloader.get_sysinfo()[0] == 'windows'