import dataclasses
import os
from typing import Any, List
import torch
from .utils import print_once
def tocsv(self):
    return [self.unique_graphs, self.captured.graphs, self.captured.operators, self.total.operators] + self.percent().tocsv()