import warnings
from typing import List, Optional, Union
import numpy as np
import PIL
import torch
from diffusers import ConfigMixin
from diffusers.image_processor import VaeImageProcessor as DiffusersVaeImageProcessor
from diffusers.utils.pil_utils import PIL_INTERPOLATION
from PIL import Image
from tqdm.auto import tqdm
class VaeImageProcessor(DiffusersVaeImageProcessor):

    @staticmethod
    def denormalize(images: np.ndarray):
        """
        Denormalize an image array to [0,1].
        """
        return np.clip(images / 2 + 0.5, 0, 1)

    def preprocess(self, image: Union[torch.FloatTensor, PIL.Image.Image, np.ndarray], height: Optional[int]=None, width: Optional[int]=None) -> np.ndarray:
        """
        Preprocess the image input. Accepted formats are PIL images, NumPy arrays or PyTorch tensors.
        """
        supported_formats = (PIL.Image.Image, np.ndarray, torch.Tensor)
        do_convert_grayscale = getattr(self.config, 'do_convert_grayscale', False)
        if do_convert_grayscale and isinstance(image, (torch.Tensor, np.ndarray)) and (image.ndim == 3):
            if isinstance(image, torch.Tensor):
                image = image.unsqueeze(1)
            elif image.shape[-1] == 1:
                image = np.expand_dims(image, axis=0)
            else:
                image = np.expand_dims(image, axis=-1)
        if isinstance(image, supported_formats):
            image = [image]
        elif not (isinstance(image, list) and all((isinstance(i, supported_formats) for i in image))):
            raise ValueError(f'Input is in incorrect format: {[type(i) for i in image]}. Currently, we only support {', '.join(supported_formats)}')
        if isinstance(image[0], PIL.Image.Image):
            if self.config.do_convert_rgb:
                image = [self.convert_to_rgb(i) for i in image]
            elif do_convert_grayscale:
                image = [self.convert_to_grayscale(i) for i in image]
            if self.config.do_resize:
                height, width = self.get_height_width(image[0], height, width)
                image = [self.resize(i, height, width) for i in image]
            image = self.reshape(self.pil_to_numpy(image))
        else:
            if isinstance(image[0], torch.Tensor):
                image = [self.pt_to_numpy(elem) for elem in image]
                image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)
            else:
                image = self.reshape(np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0))
            if do_convert_grayscale and image.ndim == 3:
                image = np.expand_dims(image, 1)
            if image.shape[1] == 4:
                return image
            if self.config.do_resize:
                height, width = self.get_height_width(image, height, width)
                image = self.resize(image, height, width)
        do_normalize = self.config.do_normalize
        if image.min() < 0 and do_normalize:
            warnings.warn(f'Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]', FutureWarning)
            do_normalize = False
        if do_normalize:
            image = self.normalize(image)
        if getattr(self.config, 'do_binarize', False):
            image = self.binarize(image)
        return image

    def postprocess(self, image: np.ndarray, output_type: str='pil', do_denormalize: Optional[List[bool]]=None):
        if not isinstance(image, np.ndarray):
            raise ValueError(f'Input for postprocessing is in incorrect format: {type(image)}. We only support np array')
        if output_type not in ['latent', 'np', 'pil']:
            deprecation_message = f'the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of these instead: `pil`, `np`, `pt`, `latent`'
            warnings.warn(deprecation_message, FutureWarning)
            output_type = 'np'
        if output_type == 'latent':
            return image
        if do_denormalize is None:
            do_denormalize = [self.config.do_normalize] * image.shape[0]
        image = np.stack([self.denormalize(image[i]) if do_denormalize[i] else image[i] for i in range(image.shape[0])], axis=0)
        image = image.transpose((0, 2, 3, 1))
        if output_type == 'pil':
            image = self.numpy_to_pil(image)
        return image

    def get_height_width(self, image: [PIL.Image.Image, np.ndarray], height: Optional[int]=None, width: Optional[int]=None):
        """
        This function return the height and width that are downscaled to the next integer multiple of
        `vae_scale_factor`.

        Args:
            image(`PIL.Image.Image`, `np.ndarray`):
                The image input, can be a PIL image, numpy array or pytorch tensor. if it is a numpy array, should have
                shape `[batch, height, width]` or `[batch, height, width, channel]` if it is a pytorch tensor, should
                have shape `[batch, channel, height, width]`.
            height (`int`, *optional*, defaults to `None`):
                The height in preprocessed image. If `None`, will use the height of `image` input.
            width (`int`, *optional*`, defaults to `None`):
                The width in preprocessed. If `None`, will use the width of the `image` input.
        """
        height = height or (image.height if isinstance(image, PIL.Image.Image) else image.shape[-2])
        width = width or (image.width if isinstance(image, PIL.Image.Image) else image.shape[-1])
        width, height = (x - x % self.config.vae_scale_factor for x in (width, height))
        return (height, width)

    @staticmethod
    def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
        """
        Convert a NumPy image to a PyTorch tensor.
        """
        if images.ndim == 3:
            images = images[..., None]
        images = torch.from_numpy(images)
        return images

    @staticmethod
    def pt_to_numpy(images: torch.FloatTensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a NumPy image.
        """
        images = images.cpu().float().numpy()
        return images

    @staticmethod
    def reshape(images: np.ndarray) -> np.ndarray:
        """
        Reshape inputs to expected shape.
        """
        if images.ndim == 3:
            images = images[..., None]
        return images.transpose(0, 3, 1, 2)

    def resize(self, image: [PIL.Image.Image, np.ndarray, torch.Tensor], height: Optional[int]=None, width: Optional[int]=None) -> [PIL.Image.Image, np.ndarray, torch.Tensor]:
        """
        Resize image.
        """
        if isinstance(image, PIL.Image.Image):
            image = image.resize((width, height), resample=PIL_INTERPOLATION[self.config.resample])
        elif isinstance(image, torch.Tensor):
            image = torch.nn.functional.interpolate(image, size=(height, width))
        elif isinstance(image, np.ndarray):
            image = self.numpy_to_pt(image)
            image = torch.nn.functional.interpolate(image, size=(height, width))
            image = self.pt_to_numpy(image)
        return image