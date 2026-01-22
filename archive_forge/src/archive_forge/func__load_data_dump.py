from abc import ABC, abstractmethod
import os
import random
from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher
from .build import build
@abstractmethod
def _load_data_dump(self):
    pass