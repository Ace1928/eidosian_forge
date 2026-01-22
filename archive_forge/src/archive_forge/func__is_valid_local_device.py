from typing import Optional, Union
import torch
@staticmethod
def _is_valid_local_device(device):
    try:
        torch.device(device)
        return True
    except Exception:
        return False