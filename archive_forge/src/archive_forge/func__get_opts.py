from collections import defaultdict
from django.apps import apps
from django.db import models
from django.db.models import Q
from django.utils.translation import gettext_lazy as _
def _get_opts(self, model, for_concrete_model):
    if for_concrete_model:
        model = model._meta.concrete_model
    return model._meta