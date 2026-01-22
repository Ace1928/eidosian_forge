import argparse
import os
from time import time
from pyzstd import compress_stream, decompress_stream, \
class ZstdTarFile(TarFile):

    def __init__(self, name, mode='r', *, level_or_option=None, zstd_dict=None, **kwargs):
        self.zstd_file = ZstdFile(name, mode, level_or_option=level_or_option, zstd_dict=zstd_dict)
        try:
            super().__init__(fileobj=self.zstd_file, mode=mode, **kwargs)
        except:
            self.zstd_file.close()
            raise

    def close(self):
        try:
            super().close()
        finally:
            self.zstd_file.close()