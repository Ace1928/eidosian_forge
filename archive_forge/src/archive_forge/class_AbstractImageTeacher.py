import copy
from typing import List, Tuple, Optional, TypeVar
from parlai.core.agents import Agent, create_agent_from_shared
from parlai.core.image_featurizers import ImageLoader
from parlai.core.loader import load_teacher_module
from parlai.core.loader import register_teacher  # noqa: F401
from parlai.core.message import Message
from parlai.core.metrics import TeacherMetrics, aggregate_named_reports
from parlai.core.opt import Opt
from parlai.utils.conversations import Conversations
from parlai.utils.data import DatatypeHelper
from parlai.utils.misc import AttrDict, no_lock, str_to_msg, warn_once
from parlai.utils.distributed import get_rank, num_workers, is_distributed
import parlai.utils.logging as logging
from abc import ABC, abstractmethod
import concurrent.futures
from threading import Thread
import queue
import random
import time
import os
import torch
import json
import argparse
class AbstractImageTeacher(FixedDialogTeacher):
    """
    Abstract class to allow easier creation of image + dialogue tasks.

    This class handles creating image features via ImageLoader if applicable
    (resnet, resnext variants) or loading existing image features from a dict
    path as per get_image_features_path().

    Important methods and properties (override in subclass if needed):

    - get_data_path(): where data file is found (default: <datapath>/<task>)
    - get_image_path(): where images found (default: <datapath>/<task>/images)
    - get_image_features_path(): dict of image features (default:
      <datapath>/<task>/image_features)
    - @property image_id_key: which key in data file objects represents image_id
    - @property text_key: which key in data file objects represents text

    Note: Assumes data files are named <dt>.json

    @abstractmethod image_id_to_image_path() must be implemented in subclass

    Example with the key defaults (but the keys can be customized):

    .. code-block:: python

        obs = {
            'text': <caption>,
            'image': <image features if specified else image>
        }
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt
        self.task = opt['task'].split(':')[1] if ':' in opt['task'] else opt['task']
        self.data_path = self.get_data_path(opt)
        self.data = self.load_data(self.data_path, self.opt)
        self.datatype = opt.get('datatype').split(':')[0]
        self._validate_image_mode_name(opt.get('image_mode'))
        self.image_mode = opt.get('image_mode')
        self.include_image = opt.get('image_mode') != 'no_image_model'
        self.image_path = self.get_image_path(opt)
        self.image_loader = None
        self.image_features_dim = opt.get('image_features_dim')
        self.blank_image_features = torch.FloatTensor(self.image_features_dim).fill_(0)
        if shared and 'data' in shared:
            self.data = shared['data']
            self.image_loader = shared['image_loader']
            if 'image_features_dict' in shared:
                self.image_features_dict = shared['image_features_dict']
        elif self.include_image:
            self.setup_image_features(self.data_path)
        else:
            warn_once('AbstractImageTeacher self.include_image was False')
            self.image_features_dict = None
        self.__verbose = False
        self.reset()

    def get_available_image_mode_names(self):
        """
        Available image model names.

        resnet and resnext variants available from the ImageLoader. resnext101_XXXXX_wsl
        is the open-sourced FB AI model (960m images, 1.5k hashtags, finetuned on
        ImageNet).
        """
        available_model_names = ImageLoader.get_available_model_names()
        return ['no_image_model', 'raw', 'ascii'] + available_model_names

    def _validate_image_mode_name(self, a):
        """
        Validate the image_mode passed in.

        Needed because image_mode used elsewhere in ParlAI is not always consistent with
        what the image teacher allows.
        """
        if not isinstance(a, str):
            raise argparse.ArgumentTypeError('%s must be a string representing image model name' % a)
        available_model_names = self.get_available_image_mode_names()
        if a not in available_model_names:
            raise argparse.ArgumentTypeError('"%s" unknown image model name. Choose from: %s. Currently suggested resnet is resnet152 and resnext is resnext101_32x48d_wsl.' % (a, available_model_names))
        return a

    @classmethod
    def add_cmdline_args(cls, argparser):
        agent = argparser.add_argument_group('AbstractImageTeacher Arguments')
        agent.add_argument('--image-path', type=str, default=None, help='Optional argument to specify where images for dataset arestored if already downloaded. Most tasks will download the imagesif not present on the < datapath > / < task > _images / * and * ifthis argument is not specified.')
        agent.add_argument('--image-features-dim', type=int, default=2048, help='Specify the size of image features Tensors.')

    @property
    def image_id_key(self):
        """
        Which key in the input data dict objects uniquely identify each image.

        Common image keys are "image_id" or "image_num". May be implemented by subclass.
        """
        return 'image_id'

    @property
    def text_key(self):
        """
        Which key in the input data dict objects identifies the text.

        Common keys are "text" or "comment". May be implemented by subclass.
        """
        return 'text'

    @abstractmethod
    def image_id_to_image_path(self, image_id):
        """
        Get the path of the image on disk.

        Must be implemented by subclass.
        """
        pass

    def get_data_path(self, opt):
        """
        Determines path to the data file.
        """
        task_name = opt['task'].split(':')[1] if ':' in opt['task'] else opt['task']
        data_path = os.path.join(opt['datapath'], task_name)
        return data_path

    def get_image_path(self, opt):
        """
        Return the path to the data directory and to the image directory.

        Is based on opt fields: task, datatype (train, valid, test), datapath.

        Subclass can override this.
        """
        data_path = self.get_data_path(opt)
        if opt.get('image_path', None):
            image_path = opt['image_path']
        else:
            image_path = os.path.join(data_path, 'images')
        return image_path

    def get_image_features_path(self, task, image_model_name, dt):
        """
        Image features for the dataset images are stored here.

        Can be overriden in subclass to use custom paths. Image features can be manually
        copied into this directory or in the case of ImageLoader eligible models, they
        will be built and stored here if not already there.
        """
        image_features_path = os.path.join(self.data_path, 'image_features')
        if not os.path.isdir(image_features_path):
            os.makedirs(image_features_path)
        return os.path.join(image_features_path, '%s_%s_%s_features_dict' % (task, image_model_name, dt))

    def is_image_mode_buildable(self, model_name):
        """
        Is buildable if features can be calculated by ImageLoader.

        Users may wish to compute features for the dataset offline and use in the model,
        in which case, the image model should return False and get_image_features()
        should be overriden in subclass.
        """
        return model_name in ImageLoader.get_available_model_names()

    def load_data(self, data_path, opt):
        """
        Loading the data file, which is the index to the images and text.

        It is often a .json file with the name of the <datatype>.json (i.e.
        train.json). Stores in self.data.

        Can be override by subclass.
        """
        dt = opt['datatype'].split(':')[0]
        if dt not in ['train', 'valid', 'val', 'test']:
            raise Exception('Unknown dt parameter: %s. Expected either "train", "valid", or "test".' % dt)
        data_file = os.path.join(self.data_path, '%s.json' % dt)
        with open(data_file, encoding='utf-8') as f:
            self.data = json.load(f)
        if len(self.data) > 0 and self.image_id_key not in self.data[0]:
            for idx, d in enumerate(self.data):
                d[self.image_id_key] = idx
        return self.data

    def setup_image_features(self, data_path):
        """
        Load text and image data.

        The image features all live in dicts by default in <data_path>/
        image_features/ but get_image_features_path() above can be overriden by
        subclass to put them elsewhere.

        In the (very odd) case that the resnet or resnext dicts (models
        buildable using ImageLoader) are not found, we build them.
        """
        if self.image_mode in ['raw', 'ascii']:
            self.image_features_dict = None
            self.image_loader = ImageLoader(self.opt)
            return
        image_mode_features_dict_path = self.get_image_features_path(self.task, self.image_mode, self.datatype)
        if os.path.isfile(image_mode_features_dict_path):
            logging.info(f'Loading existing image features dict for model: {self.image_mode} at: {image_mode_features_dict_path}')
            self.image_features_dict = torch.load(image_mode_features_dict_path, map_location='cpu')
        else:
            logging.warn('No existing image features, attempting to build.')
            if self.is_image_mode_buildable(self.image_mode):
                image_loader_opt = self.opt.copy()
                image_loader_opt['image_mode'] = self.image_mode if self.include_image else 'no_image_model'
                image_loader_opt['image_size'] = 256
                image_loader_opt['image_cropsize'] = 224
                self.image_loader = ImageLoader(image_loader_opt)
                self.image_features_dict = self._build_image_features_dict(self.data_path, self.datatype, image_mode_features_dict_path)
            else:
                raise RuntimeError('Image model: %s is not buildable by ImageLoader but doesnot already exist on disk as an image features dict forthis dataset.' % self.image_mode)

    def _build_image_features_dict(self, data_path, dt, store_dict_path):
        """
        Build resne(x)t image features with ImageLoader.

        (Or anything handleable by ImageLoader) and save to path. Only called if we
        haven't already built the dict before.
        """
        image_features_dict = {}
        total = len(self.data)
        import tqdm
        pbar = tqdm.tqdm(total=total, unit='cand', unit_scale=True, desc='Building image features dict for %s with ImageLoader.' % self.image_mode)
        num = 0
        for ex in self.data:
            img_id = ex[self.image_id_key]
            img_path = self.image_id_to_image_path(img_id)
            image = self.image_loader.load(img_path).detach()
            if 'spatial' not in self.image_mode:
                image = image[0, :, 0, 0]
            image_features_dict[img_id] = image
            num += 1
            pbar.update(1)
            if num % 1000 == 0:
                logging.debug(f'Processing image index: {num}')
        torch.save(image_features_dict, store_dict_path)
        return image_features_dict

    def reset(self):
        super().reset()
        self.example = None

    def num_episodes(self):
        return self.num_examples()

    def num_examples(self):
        return len(self.data)

    def get_image_features(self, example):
        """
        Get image features for example.

        Can be overrided in subclass for different behavior. For large datasets, it may
        be more appropriate to use the ImageLoader.load() method to load image features
        (as this is essentially streaming the features from disk, so that we do not have
        to load a large image feature dict in memory). #TODO Could be the default option
        if we are using -dt train:stream
        """
        if self.image_mode in ['raw', 'ascii']:
            try:
                image = self.image_loader.load(self.image_id_to_image_path(example['image_id']))
            except FileNotFoundError:
                image = None
            return image
        key = str(example[self.image_id_key])
        if not self.include_image or key not in self.image_features_dict:
            image_features = self.blank_image_features
        else:
            image_features = self.image_features_dict[key]
        return image_features

    def get(self, episode_idx, entry_idx=0):
        """
        Override this in subclass if your data should be handled in a different format.
        """
        example = self.data[episode_idx]
        image_features = self.get_image_features(example)
        return {'labels': [example[self.text_key]], 'image': image_features, 'episode_idx': episode_idx, 'episode_done': True}

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        shared['image_loader'] = self.image_loader
        if hasattr(self, 'image_features_dict'):
            shared['image_features_dict'] = self.image_features_dict
        return shared