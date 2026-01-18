import json
import os
import re
import numpy as np
from collections import Counter
from parlai.core.agents import Agent
from collections import defaultdict
from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from .build import build
from parlai.tasks.coco_caption.build_2014 import buildImage as buildImage_2014
from parlai.tasks.coco_caption.build_2015 import buildImage as buildImage_2015
def tokenize_mcb(self, s):
    t_str = s.lower()
    for i in ['\\?', '\\!', "\\'", '\\"', '\\$', '\\:', '\\@', '\\(', '\\)', '\\,', '\\.', '\\;']:
        t_str = re.sub(i, '', t_str)
    for i in ['\\-', '\\/']:
        t_str = re.sub(i, ' ', t_str)
    q_list = re.sub('\\?', '', t_str.lower()).split(' ')
    q_list = list(filter(lambda x: len(x) > 0, q_list))
    return q_list