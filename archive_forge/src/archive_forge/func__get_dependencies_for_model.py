import functools
import re
from collections import defaultdict
from graphlib import TopologicalSorter
from itertools import chain
from django.conf import settings
from django.db import models
from django.db.migrations import operations
from django.db.migrations.migration import Migration
from django.db.migrations.operations.models import AlterModelOptions
from django.db.migrations.optimizer import MigrationOptimizer
from django.db.migrations.questioner import MigrationQuestioner
from django.db.migrations.utils import (
def _get_dependencies_for_model(self, app_label, model_name):
    """Return foreign key dependencies of the given model."""
    dependencies = []
    model_state = self.to_state.models[app_label, model_name]
    for field in model_state.fields.values():
        if field.is_relation:
            dependencies.extend(self._get_dependencies_for_foreign_key(app_label, model_name, field, self.to_state))
    return dependencies