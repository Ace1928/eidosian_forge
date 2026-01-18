import os
import tempfile
from tempfile import TemporaryDirectory
from traitlets import Unicode
from IPython.core.application import BaseIPythonApplication
from IPython.testing import decorators as dec
@dec.onlyif_unicode_paths
def test_unicode_cwd():
    """Check that IPython starts with non-ascii characters in the path."""
    wd = tempfile.mkdtemp(suffix=u'€')
    old_wd = os.getcwd()
    os.chdir(wd)
    try:
        app = BaseIPythonApplication()
        app.init_profile_dir()
        app.init_config_files()
        app.load_config_file(suppress_errors=False)
    finally:
        os.chdir(old_wd)