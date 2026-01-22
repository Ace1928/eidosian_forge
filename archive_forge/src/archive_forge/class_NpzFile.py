import warnings
import numpy
import cupy
class NpzFile(object):

    def __init__(self, npz_file):
        self.npz_file = npz_file

    def __enter__(self):
        self.npz_file.__enter__()
        return self

    def __exit__(self, typ, val, traceback):
        self.npz_file.__exit__(typ, val, traceback)

    def __getitem__(self, key):
        arr = self.npz_file[key]
        return cupy.array(arr)

    def close(self):
        self.npz_file.close()