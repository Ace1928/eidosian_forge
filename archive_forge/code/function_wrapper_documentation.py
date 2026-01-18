import copy
import inspect
import re
from typing import (
from ..exceptions import InvalidOperationError
from ..utils.assertion import assert_or_throw
from ..utils.convert import get_full_type_path
from ..utils.entry_points import load_entry_point
from ..utils.hash import to_uuid
from .dict import IndexedOrderedDict
The decorator to register a type annotation for this function
        wrapper

        :param annotation: the type annotation
        :param code: the single char code to represent this type annotation
            , defaults to None, meaning it will try to use its parent class'
            code, this is allowed only if ``child_can_reuse_code`` is set to
            True on the parent class.
        :param matcher: a function taking in a type annotation and decide
            whether it is acceptable by the :class:`~.AnnotatedParam`
            , defaults to None, meaning it will just do a simple ``==`` check.
        :param child_can_reuse_code: whether the derived types of the current
            AnnotatedParam can reuse the code (if not specifying a new code)
            , defaults to False
        