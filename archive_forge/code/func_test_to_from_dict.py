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
def test_to_from_dict(self):
    obj = self.good_cls('Hello', 'World', 'Python')
    d = obj.as_dict()
    assert d is not None
    self.good_cls.from_dict(d)
    jsonstr = obj.to_json()
    d = json.loads(jsonstr)
    assert d['@class'], 'GoodMSONClass'
    obj = self.bad_cls('Hello', 'World')
    d = obj.as_dict()
    assert d is not None
    with pytest.raises(TypeError):
        self.bad_cls.from_dict(d)
    obj = self.bad_cls2('Hello', 'World')
    with pytest.raises(NotImplementedError):
        obj.as_dict()
    obj = self.auto_mson(2, 3)
    d = obj.as_dict()
    self.auto_mson.from_dict(d)