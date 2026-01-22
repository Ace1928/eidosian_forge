from parlai.core.teachers import DialogTeacher
from .build import build
from parlai.tasks.coco_caption.build_2014 import buildImage
from PIL import Image
import json
import os
def _image_loader(path):
    """
    Loads the appropriate image from the image_id and returns PIL Image format.
    """
    return Image.open(path).convert('RGB')