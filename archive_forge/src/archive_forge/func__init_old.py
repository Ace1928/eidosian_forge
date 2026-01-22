import warnings as _warnings
import hashlib as _hashlib
def _init_old(self, key, msg, digestmod):
    if callable(digestmod):
        digest_cons = digestmod
    elif isinstance(digestmod, str):
        digest_cons = lambda d=b'': _hashlib.new(digestmod, d)
    else:
        digest_cons = lambda d=b'': digestmod.new(d)
    self._hmac = None
    self._outer = digest_cons()
    self._inner = digest_cons()
    self.digest_size = self._inner.digest_size
    if hasattr(self._inner, 'block_size'):
        blocksize = self._inner.block_size
        if blocksize < 16:
            _warnings.warn('block_size of %d seems too small; using our default of %d.' % (blocksize, self.blocksize), RuntimeWarning, 2)
            blocksize = self.blocksize
    else:
        _warnings.warn('No block_size attribute on given digest object; Assuming %d.' % self.blocksize, RuntimeWarning, 2)
        blocksize = self.blocksize
    if len(key) > blocksize:
        key = digest_cons(key).digest()
    self.block_size = blocksize
    key = key.ljust(blocksize, b'\x00')
    self._outer.update(key.translate(trans_5C))
    self._inner.update(key.translate(trans_36))
    if msg is not None:
        self.update(msg)