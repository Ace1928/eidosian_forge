import parlai.core.build_data as build_data
import parlai.utils.logging as logging
import os
from PIL import Image
from zipfile import ZipFile
def _lazy_import_torch(self):
    try:
        import torch
    except ImportError:
        raise ImportError('Need to install Pytorch: go to pytorch.org')
    import torchvision
    import torchvision.transforms as transforms
    import torch.nn as nn
    self.use_cuda = not self.opt.get('no_cuda', False) and torch.cuda.is_available()
    if self.use_cuda:
        logging.debug(f'Using CUDA')
        torch.cuda.set_device(self.opt.get('gpu', -1))
    self.torch = torch
    self.torchvision = torchvision
    self.transforms = transforms
    self.nn = nn