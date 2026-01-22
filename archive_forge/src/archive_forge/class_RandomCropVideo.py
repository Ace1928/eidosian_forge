import numbers
import random
import warnings
from torchvision.transforms import RandomCrop, RandomResizedCrop
from . import _functional_video as F
class RandomCropVideo(RandomCrop):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: randomly cropped/resized video clip.
                size is (C, T, OH, OW)
        """
        i, j, h, w = self.get_params(clip, self.size)
        return F.crop(clip, i, j, h, w)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'