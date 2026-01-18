import shutil
import os
@classmethod
def tmp_folder(cls, name=''):
    cls.tmp_folders.add(name)
    return name