from parlai.core.teachers import FixedDialogTeacher
from .build import build
import os
import json
class CCPEAssistantTeacher(CCPEAllTeacher):

    def _setup_data(self):
        super()._setup_data()
        self.data = self.assistantData