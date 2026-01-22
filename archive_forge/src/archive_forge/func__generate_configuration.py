import os
import shutil
import tempfile
import pyomo.common.envvar as envvar
from pyomo.common.fileutils import this_file_dir, find_dir
from pyomo.common.download import FileDownloader
def _generate_configuration():
    from setuptools.extension import Extension
    pathlist = [os.path.join(envvar.PYOMO_CONFIG_DIR, 'src'), this_file_dir()]
    if 'MCPP_ROOT' in os.environ:
        mcpp = os.environ['MCPP_ROOT']
    else:
        mcpp = find_dir('mcpp', cwd=True, pathlist=pathlist)
    if mcpp:
        print('Found MC++ at %s' % (mcpp,))
    else:
        raise RuntimeError('Cannot identify the location of the MCPP source distribution')
    project_dir = this_file_dir()
    sources = [os.path.join(project_dir, 'mcppInterface.cpp')]
    include_dirs = [os.path.join(mcpp, 'src', 'mc'), os.path.join(mcpp, 'src', '3rdparty', 'fadbad++')]
    mcpp_ext = Extension('mcppInterface', sources=sources, language='c++', extra_compile_args=[], include_dirs=include_dirs, library_dirs=[], libraries=[])
    package_config = {'name': 'mcpp', 'packages': [], 'ext_modules': [mcpp_ext]}
    return package_config