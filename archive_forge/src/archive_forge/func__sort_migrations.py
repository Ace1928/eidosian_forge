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
def _sort_migrations(self):
    """
        Reorder to make things possible. Reordering may be needed so FKs work
        nicely inside the same app.
        """
    for app_label, ops in sorted(self.generated_operations.items()):
        ts = TopologicalSorter()
        for op in ops:
            ts.add(op)
            for dep in op._auto_deps:
                dep = self._resolve_dependency(dep)[0]
                if dep[0] != app_label:
                    continue
                ts.add(op, *(x for x in ops if self.check_dependency(x, dep)))
        self.generated_operations[app_label] = list(ts.static_order())