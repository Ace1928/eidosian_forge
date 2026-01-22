from ._base import *
from .models import LazyDBCacheBase
class PklDBCache(LazyDBCacheBase):

    def dumps(self, data, *args, **kwargs):
        return pkler.dumps(data, *args, **kwargs)

    def loads(self, data, *args, **kwargs):
        return pkler.loads(data, *args, **kwargs)

    def restore(self, *args, **kwargs):
        self.lock.acquire()
        dbdata = {}
        with self.lock:
            try:
                dbdata = fio.pklload(self.cache_filepath)
            except Exception as e:
                logger.error(f'Failed to Restore DB {self._cachename}: {str(e)}')
                logger.error(f'Copying DB to Backup')
                tempfn = fio.mod_fname(self.cache_filepath, suffix='_' + tstamp())
                fio.copy(self.cache_filepath, tempfn)
                logger.error(f'DB saved to {tempfn}')
        self.lock.release()
        return dbdata

    def save(self, dbdata, *args, **kwargs):
        _saved = False
        self.lock.acquire()
        with self.lock:
            try:
                fio.pklsave(dbdata, self.cache_filepath)
                _saved = True
            except Exception as e:
                logger.error(f'Failed to Save DB {self._cachename}: {str(e)}')
                _saved = False
        self.lock.release()
        return _saved