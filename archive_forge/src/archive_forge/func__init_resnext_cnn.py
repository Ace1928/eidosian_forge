import parlai.core.build_data as build_data
import parlai.utils.logging as logging
import os
from PIL import Image
from zipfile import ZipFile
def _init_resnext_cnn(self):
    """
        Lazily initialize preprocessor model.

        When image_mode is one of the ``resnext101_..._wsl`` varieties
        """
    try:
        cnn_type, layer_num = self._image_mode_switcher()
        model = self.torch.hub.load('facebookresearch/WSL-Images', cnn_type)
        self.netCNN = self.nn.Sequential(*list(model.children())[:layer_num])
    except RuntimeError as e:
        model_names = [m for m in IMAGE_MODE_SWITCHER if 'resnext101' in m]
        logging.error(f'If you have specified one of the resnext101 wsl models, please make sure it is one of the following: \n{', '.join(model_names)}')
        raise e
    except AttributeError:
        raise RuntimeError('Please install the latest pytorch distribution to have access to the resnext101 wsl models (pytorch 1.1.0, torchvision 0.3.0)')
    if self.use_cuda:
        self.netCNN.cuda()