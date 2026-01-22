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