import bz2
import re
from io import BytesIO
import fastbencode as bencode
from .... import errors, iterablefile, lru_cache, multiparent, osutils
from .... import repository as _mod_repository
from .... import revision as _mod_revision
from .... import trace, ui
from ....i18n import ngettext
from ... import pack, serializer
from ... import versionedfile as _mod_versionedfile
from .. import bundle_data
from .. import serializer as bundle_serializer
def _install_mp_records_keys(self, versionedfile, records):
    d_func = multiparent.MultiParent.from_patch
    vf_records = []
    for key, meta, text in records:
        if len(key) == 2:
            prefix = key[:1]
        else:
            prefix = ()
        parents = [prefix + (parent,) for parent in meta[b'parents']]
        vf_records.append((key, parents, meta[b'sha1'], d_func(text)))
    versionedfile.add_mpdiffs(vf_records)