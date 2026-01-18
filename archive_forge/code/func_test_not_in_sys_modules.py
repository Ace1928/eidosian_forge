import types
from pathlib import Path
import pytest
from tempfile import TemporaryDirectory
from IPython.lib.deepreload import modules_reloading
from IPython.lib.deepreload import reload as dreload
from IPython.utils.syspathcontext import prepended_to_syspath
def test_not_in_sys_modules():
    fake_module = types.ModuleType('fake_module')
    with pytest.raises(ImportError, match='not in sys.modules'):
        dreload(fake_module)