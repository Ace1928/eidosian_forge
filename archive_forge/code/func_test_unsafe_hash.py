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
def test_unsafe_hash(self):
    GMC = GoodMSONClass
    a_list = [GMC(1, 1.0, 'one'), GMC(2, 2.0, 'two')]
    b_dict = {'first': GMC(3, 3.0, 'three'), 'second': GMC(4, 4.0, 'four')}
    c_list_dict_list = [{'list1': [GMC(5, 5.0, 'five'), GMC(6, 6.0, 'six'), GMC(7, 7.0, 'seven')], 'list2': [GMC(8, 8.0, 'eight')]}, {'list3': [GMC(9, 9.0, 'nine'), GMC(10, 10.0, 'ten'), GMC(11, 11.0, 'eleven'), GMC(12, 12.0, 'twelve')], 'list4': [GMC(13, 13.0, 'thirteen'), GMC(14, 14.0, 'fourteen')], 'list5': [GMC(15, 15.0, 'fifteen')]}]
    obj = GoodNestedMSONClass(a_list=a_list, b_dict=b_dict, c_list_dict_list=c_list_dict_list)
    assert a_list[0].unsafe_hash().hexdigest() == 'ea44de0e2ef627be582282c02c48e94de0d58ec6'
    assert obj.unsafe_hash().hexdigest() == '44204c8da394e878f7562c9aa2e37c2177f28b81'