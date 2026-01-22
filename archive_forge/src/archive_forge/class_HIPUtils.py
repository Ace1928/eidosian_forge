import abc
import hashlib
import os
import tempfile
from pathlib import Path
from ..common.build import _build
from .cache import get_cache_manager
class HIPUtils(object):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(HIPUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, 'backends', 'hip.c')).read_text()
        key = hashlib.md5(src.encode('utf-8')).hexdigest()
        cache = get_cache_manager(key)
        fname = 'hip_utils.so'
        cache_path = cache.get_file(fname)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, 'main.c')
                with open(src_path, 'w') as f:
                    f.write(src)
                so = _build('hip_utils', src_path, tmpdir)
                with open(so, 'rb') as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location('hip_utils', cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties