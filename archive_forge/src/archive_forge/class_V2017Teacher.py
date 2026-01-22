from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from .build_2014 import build as build_2014
from .build_2014 import buildImage as buildImage_2014
from .build_2017 import build as build_2017
from .build_2017 import buildImage as buildImage_2017
import os
import json
import random
class V2017Teacher(DefaultTeacher):

    def __init__(self, opt, shared=None):
        super(V2017Teacher, self).__init__(opt, shared, '2017')