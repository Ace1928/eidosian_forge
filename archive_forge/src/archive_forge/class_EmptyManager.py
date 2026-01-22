import copy
import inspect
from functools import wraps
from importlib import import_module
from django.db import router
from django.db.models.query import QuerySet
class EmptyManager(Manager):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_queryset(self):
        return super().get_queryset().none()