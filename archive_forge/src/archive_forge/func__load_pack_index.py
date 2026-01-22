import posixpath
import tempfile
from ..object_store import BucketBasedObjectStore
from ..pack import PACK_SPOOL_FILE_MAX_SIZE, Pack, PackData, load_pack_index_file
def _load_pack_index(self, name):
    b = self.bucket.blob(posixpath.join(self.subpath, name + '.idx'))
    f = tempfile.SpooledTemporaryFile(max_size=PACK_SPOOL_FILE_MAX_SIZE)
    b.download_to_file(f)
    f.seek(0)
    return load_pack_index_file(name + '.idx', f)