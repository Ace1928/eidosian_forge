import weakref
from unittest import mock
from cliff import lister
from cliff.tests import base
class ExerciseListerDifferentTypes(ExerciseLister):
    data = ExerciseLister.data + [(1, 0)]