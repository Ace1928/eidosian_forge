import datetime
import os
import stat
from contextlib import contextmanager
from io import BytesIO
from itertools import permutations
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import (
from .utils import ext_functest_builder, functest_builder, make_commit, make_object
def test_serialize_mergetag(self):
    tag = make_object(Tag, object=(Commit, b'a38d6181ff27824c79fc7df825164a212eff6a3f'), object_type_name=b'commit', name=b'v2.6.22-rc7', tag_time=1183319674, tag_timezone=0, tagger=b'Linus Torvalds <torvalds@woody.linux-foundation.org>', message=default_message)
    commit = self.make_commit(mergetag=[tag])
    self.assertEqual(b'tree d80c186a03f423a81b39df39dc87fd269736ca86\nparent ab64bbdcc51b170d21588e5c5d391ee5c0c96dfd\nparent 4cffe90e0a41ad3f5190079d7c8f036bde29cbe6\nauthor James Westby <jw+debian@jameswestby.net> 1174773719 +0000\ncommitter James Westby <jw+debian@jameswestby.net> 1174773719 +0000\nmergetag object a38d6181ff27824c79fc7df825164a212eff6a3f\n type commit\n tag v2.6.22-rc7\n tagger Linus Torvalds <torvalds@woody.linux-foundation.org> 1183319674 +0000\n \n Linux 2.6.22-rc7\n -----BEGIN PGP SIGNATURE-----\n Version: GnuPG v1.4.7 (GNU/Linux)\n \n iD8DBQBGiAaAF3YsRnbiHLsRAitMAKCiLboJkQECM/jpYsY3WPfvUgLXkACgg3ql\n OK2XeQOiEeXtT76rV4t2WR4=\n =ivrA\n -----END PGP SIGNATURE-----\n\nMerge ../b\n', commit.as_raw_string())