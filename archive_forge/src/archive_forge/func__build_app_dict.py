from functools import update_wrapper
from weakref import WeakSet
from django.apps import apps
from django.conf import settings
from django.contrib.admin import ModelAdmin, actions
from django.contrib.admin.exceptions import AlreadyRegistered, NotRegistered
from django.contrib.admin.views.autocomplete import AutocompleteJsonView
from django.contrib.auth import REDIRECT_FIELD_NAME
from django.core.exceptions import ImproperlyConfigured
from django.db.models.base import ModelBase
from django.http import Http404, HttpResponsePermanentRedirect, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import NoReverseMatch, Resolver404, resolve, reverse
from django.utils.decorators import method_decorator
from django.utils.functional import LazyObject
from django.utils.module_loading import import_string
from django.utils.text import capfirst
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy
from django.views.decorators.cache import never_cache
from django.views.decorators.common import no_append_slash
from django.views.decorators.csrf import csrf_protect
from django.views.i18n import JavaScriptCatalog
def _build_app_dict(self, request, label=None):
    """
        Build the app dictionary. The optional `label` parameter filters models
        of a specific app.
        """
    app_dict = {}
    if label:
        models = {m: m_a for m, m_a in self._registry.items() if m._meta.app_label == label}
    else:
        models = self._registry
    for model, model_admin in models.items():
        app_label = model._meta.app_label
        has_module_perms = model_admin.has_module_permission(request)
        if not has_module_perms:
            continue
        perms = model_admin.get_model_perms(request)
        if True not in perms.values():
            continue
        info = (app_label, model._meta.model_name)
        model_dict = {'model': model, 'name': capfirst(model._meta.verbose_name_plural), 'object_name': model._meta.object_name, 'perms': perms, 'admin_url': None, 'add_url': None}
        if perms.get('change') or perms.get('view'):
            model_dict['view_only'] = not perms.get('change')
            try:
                model_dict['admin_url'] = reverse('admin:%s_%s_changelist' % info, current_app=self.name)
            except NoReverseMatch:
                pass
        if perms.get('add'):
            try:
                model_dict['add_url'] = reverse('admin:%s_%s_add' % info, current_app=self.name)
            except NoReverseMatch:
                pass
        if app_label in app_dict:
            app_dict[app_label]['models'].append(model_dict)
        else:
            app_dict[app_label] = {'name': apps.get_app_config(app_label).verbose_name, 'app_label': app_label, 'app_url': reverse('admin:app_list', kwargs={'app_label': app_label}, current_app=self.name), 'has_module_perms': has_module_perms, 'models': [model_dict]}
    return app_dict