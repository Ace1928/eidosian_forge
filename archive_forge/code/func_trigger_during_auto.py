from stat import S_ISDIR
from ... import controldir, errors, gpg, osutils, repository
from ... import revision as _mod_revision
from ... import tests, transport, ui
from ...tests import TestCaseWithTransport, TestNotApplicable, test_server
from ...transport import memory
from .. import inventory
from ..btree_index import BTreeGraphIndex
from ..groupcompress_repo import RepositoryFormat2a
from ..index import GraphIndex
from ..smart import client
def trigger_during_auto(*args, **kwargs):
    ret = orig(*args, **kwargs)
    if not autopack_count[0]:
        r2.pack()
    autopack_count[0] += 1
    return ret