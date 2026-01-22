from parlai.core.teachers import FbDialogTeacher, FixedDialogTeacher
from .build import build
import copy
import os
class LmTeacher(FunpediaTeacher):
    """
    Modifies the data to drop the query entirely, creating a language modeling task.
    """

    def get_text(self, entry):
        return ''