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
def test_objectid(self):
    oid = ObjectId('562e8301218dcbbc3d7d91ce')
    with pytest.raises(TypeError):
        json.dumps(oid)
    djson = json.dumps(oid, cls=MontyEncoder)
    x = json.loads(djson, cls=MontyDecoder)
    assert isinstance(x, ObjectId)