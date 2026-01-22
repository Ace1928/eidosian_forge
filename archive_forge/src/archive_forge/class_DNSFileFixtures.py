import os
from libcloud.utils.py3 import u
class DNSFileFixtures(FileFixtures):

    def __init__(self, sub_dir=''):
        super().__init__(fixtures_type='dns', sub_dir=sub_dir)