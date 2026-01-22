import json
import math
import re
import struct
import sys
from peewee import *
from peewee import ColumnBase
from peewee import EnclosedNodeList
from peewee import Entity
from peewee import Expression
from peewee import Insert
from peewee import Node
from peewee import NodeList
from peewee import OP
from peewee import VirtualField
from peewee import merge_dict
from peewee import sqlite3
def _json_contains(src_json, obj_json):
    stack = []
    try:
        stack.append((json.loads(obj_json), json.loads(src_json)))
    except:
        return False
    while stack:
        obj, src = stack.pop()
        if isinstance(src, dict):
            if isinstance(obj, dict):
                for key in obj:
                    if key not in src:
                        return False
                    stack.append((obj[key], src[key]))
            elif isinstance(obj, list):
                for item in obj:
                    if item not in src:
                        return False
            elif obj not in src:
                return False
        elif isinstance(src, list):
            if isinstance(obj, dict):
                return False
            elif isinstance(obj, list):
                try:
                    for i in range(len(obj)):
                        stack.append((obj[i], src[i]))
                except IndexError:
                    return False
            elif obj not in src:
                return False
        elif obj != src:
            return False
    return True