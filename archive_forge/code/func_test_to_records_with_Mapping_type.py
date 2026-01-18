from collections import abc
import email
from email.parser import Parser
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_records_with_Mapping_type(self):
    abc.Mapping.register(email.message.Message)
    headers = Parser().parsestr('From: <user@example.com>\nTo: <someone_else@example.com>\nSubject: Test message\n\nBody would go here\n')
    frame = DataFrame.from_records([headers])
    all((x in frame for x in ['Type', 'Subject', 'From']))