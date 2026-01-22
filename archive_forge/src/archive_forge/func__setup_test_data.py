from parlai.core.teachers import (
from parlai.core.opt import Opt
import copy
import random
import itertools
import os
from PIL import Image
import string
import json
from abc import ABC
from typing import Tuple, List
def _setup_test_data(self, opt):
    datapath = os.path.join(opt['datapath'], 'ImageTeacher')
    imagepath = os.path.join(datapath, 'images')
    os.makedirs(imagepath, exist_ok=True)
    self.image_features_path = os.path.join(datapath, f'{opt['image_mode']}_image_features')
    imgs = [f'img_{i}' for i in range(10)]
    for i, img in enumerate(imgs):
        image = Image.new('RGB', (16, 16), color=i)
        image.save(os.path.join(imagepath, f'{img}.jpg'), 'JPEG')
    for dt in ['train', 'valid', 'test']:
        random.seed(42)
        data = [{'image_id': img, 'text': string.ascii_uppercase[i]} for i, img in enumerate(imgs)]
        with open(os.path.join(datapath, f'{dt}.json'), 'w') as f:
            json.dump(data, f)