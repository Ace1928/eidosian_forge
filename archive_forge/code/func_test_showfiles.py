from .. import utils
import builtins
import io
import logging
import os
import pytest
import tempfile
import sys
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import callbacks
from rpy2.rinterface_lib import openrlib
@pytest.mark.skipif(os.name == 'nt', reason='Not supported on Windows')
def test_showfiles():
    sf = []

    def f(filenames, headers, wtitle, pager):
        sf.append(wtitle)
        for tf in filenames:
            sf.append(tf)
    with utils.obj_in_module(callbacks, 'showfiles', f):
        file_path = rinterface.baseenv['file.path']
        r_home = rinterface.baseenv['R.home']
        filename = file_path(r_home(rinterface.StrSexpVector(('doc',))), rinterface.StrSexpVector(('COPYRIGHTS',)))
        rinterface.baseenv['file.show'](filename)
        assert filename[0] == sf[1]
        assert 'R Information' == sf[0]