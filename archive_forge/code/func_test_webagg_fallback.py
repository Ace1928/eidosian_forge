import subprocess
import os
import sys
import pytest
import matplotlib.backends.backend_webagg_core
@pytest.mark.parametrize('backend', ['webagg', 'nbagg'])
def test_webagg_fallback(backend):
    pytest.importorskip('tornado')
    if backend == 'nbagg':
        pytest.importorskip('IPython')
    env = dict(os.environ)
    if sys.platform != 'win32':
        env['DISPLAY'] = ''
    env['MPLBACKEND'] = backend
    test_code = 'import os;' + f"assert os.environ['MPLBACKEND'] == '{backend}';" + 'import matplotlib.pyplot as plt; ' + f"print(plt.get_backend());assert '{backend}' == plt.get_backend().lower();"
    ret = subprocess.call([sys.executable, '-c', test_code], env=env)
    assert ret == 0