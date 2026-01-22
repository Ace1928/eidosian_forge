import os
import shutil
import tempfile
import pyomo.common.envvar as envvar
from pyomo.common.fileutils import this_file_dir, find_dir
from pyomo.common.download import FileDownloader
class _BuildWithoutPlatformInfo(build_ext, object):

    def get_ext_filename(self, ext_name):
        filename = super(_BuildWithoutPlatformInfo, self).get_ext_filename(ext_name).split('.')
        filename = '.'.join([filename[0], filename[-1]])
        return filename