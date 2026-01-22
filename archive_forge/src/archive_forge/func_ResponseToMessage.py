from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
def ResponseToMessage(response):
    return {'text': {'text': [response]}}