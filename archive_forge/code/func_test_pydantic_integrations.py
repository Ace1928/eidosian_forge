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
def test_pydantic_integrations(self):
    from pydantic import BaseModel
    global ModelWithMSONable

    class ModelWithMSONable(BaseModel):
        a: GoodMSONClass
    test_object = ModelWithMSONable(a=GoodMSONClass(1, 1, 1))
    test_dict_object = ModelWithMSONable(a=test_object.a.as_dict())
    assert test_dict_object.a.a == test_object.a.a
    assert test_object.model_json_schema() == {'title': 'ModelWithMSONable', 'type': 'object', 'properties': {'a': {'title': 'A', 'type': 'object', 'properties': {'@class': {'enum': ['GoodMSONClass'], 'type': 'string'}, '@module': {'enum': ['tests.test_json'], 'type': 'string'}, '@version': {'type': 'string'}}, 'required': ['@class', '@module']}}, 'required': ['a']}
    d = jsanitize(test_object, strict=True, enum_values=True, allow_bson=True)
    assert d == {'a': {'@module': 'tests.test_json', '@class': 'GoodMSONClass', '@version': '0.1', 'a': 1, 'b': 1, 'c': 1, 'd': 1, 'values': []}, '@module': 'tests.test_json', '@class': 'ModelWithMSONable', '@version': '0.1'}
    obj = MontyDecoder().process_decoded(d)
    assert isinstance(obj, BaseModel)
    assert isinstance(obj.a, GoodMSONClass)
    assert obj.a.b == 1