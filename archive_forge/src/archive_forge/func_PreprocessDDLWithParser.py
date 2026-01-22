from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
from googlecloudsdk.core import exceptions
def PreprocessDDLWithParser(ddl_text):
    return DDLParser(ddl_text).Process()