from __future__ import absolute_import
import functools
import json
import re
import sys
def __put_value(self, value):
    top = self.stack[-1]
    if isinstance(top, dict):
        top[self.member_name] = value
    else:
        top.append(value)