import weakref
from unittest import mock
from cliff import lister
from cliff.tests import base
class ExerciseListerNullValues(ExerciseLister):
    data = ExerciseLister.data + [(None, None)]