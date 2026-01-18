import os
import tempfile
from tempfile import TemporaryDirectory
from traitlets import Unicode
from IPython.core.application import BaseIPythonApplication
from IPython.testing import decorators as dec
@dec.onlyif_unicode_paths
def test_unicode_ipdir():
    """Check that IPython starts with non-ascii characters in the IP dir."""
    ipdir = tempfile.mkdtemp(suffix=u'€')
    with open(os.path.join(ipdir, 'ipython_config.py'), 'w', encoding='utf-8') as f:
        pass
    old_ipdir1 = os.environ.pop('IPYTHONDIR', None)
    old_ipdir2 = os.environ.pop('IPYTHON_DIR', None)
    os.environ['IPYTHONDIR'] = ipdir
    try:
        app = BaseIPythonApplication()
        app.init_profile_dir()
        app.init_config_files()
        app.load_config_file(suppress_errors=False)
    finally:
        if old_ipdir1:
            os.environ['IPYTHONDIR'] = old_ipdir1
        if old_ipdir2:
            os.environ['IPYTHONDIR'] = old_ipdir2