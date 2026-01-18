from typing import List
import torch
from torch._export.db.case import export_case

    Lists are treated as static construct, therefore unpacking should be
    erased after tracing.
    