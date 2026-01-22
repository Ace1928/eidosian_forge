import copy
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from django.apps import AppConfig
from django.apps.registry import Apps
from django.apps.registry import apps as global_apps
from django.conf import settings
from django.core.exceptions import FieldDoesNotExist
from django.db import models
from django.db.migrations.utils import field_is_referenced, get_references
from django.db.models import NOT_PROVIDED
from django.db.models.fields.related import RECURSIVE_RELATIONSHIP_CONSTANT
from django.db.models.options import DEFAULT_NAMES, normalize_together
from django.db.models.utils import make_model_tuple
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from django.utils.version import get_docs_version
from .exceptions import InvalidBasesError
from .utils import resolve_relation
class StateApps(Apps):
    """
    Subclass of the global Apps registry class to better handle dynamic model
    additions and removals.
    """

    def __init__(self, real_apps, models, ignore_swappable=False):
        self.real_models = []
        for app_label in real_apps:
            app = global_apps.get_app_config(app_label)
            for model in app.get_models():
                self.real_models.append(ModelState.from_model(model, exclude_rels=True))
        app_labels = {model_state.app_label for model_state in models.values()}
        app_configs = [AppConfigStub(label) for label in sorted([*real_apps, *app_labels])]
        super().__init__(app_configs)
        self._lock = None
        self.ready_event = None
        self.render_multiple([*models.values(), *self.real_models])
        from django.core.checks.model_checks import _check_lazy_references
        ignore = {make_model_tuple(settings.AUTH_USER_MODEL)} if ignore_swappable else set()
        errors = _check_lazy_references(self, ignore=ignore)
        if errors:
            raise ValueError('\n'.join((error.msg for error in errors)))

    @contextmanager
    def bulk_update(self):
        ready = self.ready
        self.ready = False
        try:
            yield
        finally:
            self.ready = ready
            self.clear_cache()

    def render_multiple(self, model_states):
        if not model_states:
            return
        with self.bulk_update():
            unrendered_models = model_states
            while unrendered_models:
                new_unrendered_models = []
                for model in unrendered_models:
                    try:
                        model.render(self)
                    except InvalidBasesError:
                        new_unrendered_models.append(model)
                if len(new_unrendered_models) == len(unrendered_models):
                    raise InvalidBasesError('Cannot resolve bases for %r\nThis can happen if you are inheriting models from an app with migrations (e.g. contrib.auth)\n in an app with no migrations; see https://docs.djangoproject.com/en/%s/topics/migrations/#dependencies for more' % (new_unrendered_models, get_docs_version()))
                unrendered_models = new_unrendered_models

    def clone(self):
        """Return a clone of this registry."""
        clone = StateApps([], {})
        clone.all_models = copy.deepcopy(self.all_models)
        for app_label in self.app_configs:
            app_config = AppConfigStub(app_label)
            app_config.apps = clone
            app_config.import_models()
            clone.app_configs[app_label] = app_config
        clone.real_models = self.real_models
        return clone

    def register_model(self, app_label, model):
        self.all_models[app_label][model._meta.model_name] = model
        if app_label not in self.app_configs:
            self.app_configs[app_label] = AppConfigStub(app_label)
            self.app_configs[app_label].apps = self
        self.app_configs[app_label].models[model._meta.model_name] = model
        self.do_pending_operations(model)
        self.clear_cache()

    def unregister_model(self, app_label, model_name):
        try:
            del self.all_models[app_label][model_name]
            del self.app_configs[app_label].models[model_name]
        except KeyError:
            pass