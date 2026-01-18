import re
from pytest import raises
from ...types import Argument, Field, Int, List, NonNull, ObjectType, Schema, String
from ..connection import (
from ..node import Node
def test_pageinfo():
    assert PageInfo._meta.name == 'PageInfo'
    fields = PageInfo._meta.fields
    assert list(fields) == ['has_next_page', 'has_previous_page', 'start_cursor', 'end_cursor']