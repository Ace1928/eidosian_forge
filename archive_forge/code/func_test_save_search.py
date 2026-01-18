from uuid import uuid1
from pyxnat import Interface
from pyxnat import jsonutil
import os.path as op
def test_save_search():
    central.manage.search.save(search_name, 'xnat:mrSessionData', central.inspect.datatypes('xnat:mrSessionData'), [('xnat:mrSessionData/XNAT_COL_MRSESSIONDATAFIELDSTRENGTH', 'LIKE', '*1.5*'), 'AND'])
    assert search_name in central.manage.search.saved()