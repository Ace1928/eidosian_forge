from enum import Enum
import torch
from torch._export.db.case import export_case
class Animal(Enum):
    COW = 'moo'