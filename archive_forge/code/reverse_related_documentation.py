import warnings
from django.core import exceptions
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
from . import BLANK_CHOICE_DASH
from .mixins import FieldCacheMixin

        Return the field in the 'to' object to which this relationship is tied.
        Provided for symmetry with ManyToOneRel.
        