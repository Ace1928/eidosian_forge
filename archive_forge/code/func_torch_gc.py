import os
import sysconfig
def torch_gc():
    try:
        import torch
        if torch.cuda.is_available():
            with torch.cuda.device(0):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    except:
        pass