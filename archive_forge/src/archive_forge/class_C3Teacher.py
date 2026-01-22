from .build import build
from parlai.tasks.dream.agents import DREAMTeacher
import os
class C3Teacher(DREAMTeacher):

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'c3'

    def setup_data(self):
        build(self.opt)
        jsons_path = os.path.join(self.opt['datapath'], 'C3')
        return self.setup_helper(jsons_path)