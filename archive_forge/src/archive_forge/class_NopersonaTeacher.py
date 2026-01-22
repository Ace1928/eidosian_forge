from parlai.core.teachers import FbDialogTeacher, FixedDialogTeacher
from .build import build
import copy
import os
class NopersonaTeacher(FunpediaTeacher):
    """
    Strips persona out entirely.
    """

    def get_text(self, entry):
        return entry['title'] + '\n' + entry['passage']