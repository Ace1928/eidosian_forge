import base64
import io
from typing import Dict, Union
import PIL
import torch
from parlai.core.image_featurizers import ImageLoader
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.utils.typing import TShared
class ImageFeaturesGenerator(object):
    """
    Features generator for images.

    Uses ParlAI Image Loader.
    """

    def __init__(self, opt: Opt, shared: TShared=None):
        self.opt = opt
        self.image_model = opt.get('image_mode')
        if shared:
            self.image_loader = shared['image_loader']
        else:
            opt.setdefault('image_mode', self.image_model)
            new_opt = ParlaiParser(True, False).parse_args([])
            for k, v in new_opt.items():
                if k not in opt:
                    opt[k] = v
            self.image_loader = ImageLoader(opt)

    def get_image_features(self, image_id: str, image: 'PIL.Image') -> torch.Tensor:
        """
        Get image features for given image id and Image.

        :param image_id:
            id for image
        :param image:
            PIL Image object

        :return image_features:
            Image Features Tensor
        """
        image = image.convert('RGB')
        return self.image_loader.extract(image)