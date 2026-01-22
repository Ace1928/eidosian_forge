import os
from libcloud.utils.py3 import u
class StorageFileFixtures(FileFixtures):

    def __init__(self, sub_dir=''):
        super().__init__(fixtures_type='storage', sub_dir=sub_dir)