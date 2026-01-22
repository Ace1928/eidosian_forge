from __future__ import annotations
import dataclasses
import datetime
import json
import os
import pathlib
from enum import Enum
import numpy as np
import pandas as pd
import pytest
import torch
from bson.objectid import ObjectId
from monty.json import MontyDecoder, MontyEncoder, MSONable, _load_redirect, jsanitize
from . import __version__ as tests_version
class GoodNestedMSONClass(MSONable):

    def __init__(self, a_list, b_dict, c_list_dict_list, **kwargs):
        assert isinstance(a_list, list)
        assert isinstance(b_dict, dict)
        assert isinstance(c_list_dict_list, list)
        assert isinstance(c_list_dict_list[0], dict)
        first_key = list(c_list_dict_list[0].keys())[0]
        assert isinstance(c_list_dict_list[0][first_key], list)
        self.a_list = a_list
        self.b_dict = b_dict
        self._c_list_dict_list = c_list_dict_list
        self.kwargs = kwargs