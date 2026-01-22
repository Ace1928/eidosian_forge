import base64
import io
from typing import Dict, Union
import PIL
import torch
from parlai.core.image_featurizers import ImageLoader
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.utils.typing import TShared
class ImageInformation(object):
    """
    Representation of image information.
    """

    def __init__(self, image_id: str, image_location_id: str, image: Union['PIL.Image', str]):
        """
        When image is str, it is a serialized; need to deserialize.
        """
        self._image_id = image_id
        self._image_location_id = image_location_id
        self._image = image
        if isinstance(self._image, str):
            self._image = PIL.Image.open(io.BytesIO(base64.b64decode(self._image)))

    def get_image_id(self) -> str:
        return self._image_id

    def get_image_location_id(self) -> str:
        return self._image_location_id

    def get_image(self) -> 'PIL.Image':
        return self._image

    def offload_state(self) -> Dict[str, str]:
        """
        Return serialized state.

        :return state_dict:
            serialized state that can be used in json.dumps
        """
        byte_arr = io.BytesIO()
        image = self.get_image()
        image.save(byte_arr, format='JPEG')
        serialized = base64.encodebytes(byte_arr.getvalue()).decode('utf-8')
        return {'image_id': self.get_image_id(), 'image_location_id': self.get_image_location_id(), 'image': serialized}