import copy
import enum
import json
import re
from functools import partial, update_wrapper
from urllib.parse import parse_qsl
from urllib.parse import quote as urlquote
from urllib.parse import urlparse
from django import forms
from django.conf import settings
from django.contrib import messages
from django.contrib.admin import helpers, widgets
from django.contrib.admin.checks import (
from django.contrib.admin.exceptions import DisallowedModelAdminToField, NotRegistered
from django.contrib.admin.templatetags.admin_urls import add_preserved_filters
from django.contrib.admin.utils import (
from django.contrib.admin.widgets import AutocompleteSelect, AutocompleteSelectMultiple
from django.contrib.auth import get_permission_codename
from django.core.exceptions import (
from django.core.paginator import Paginator
from django.db import models, router, transaction
from django.db.models.constants import LOOKUP_SEP
from django.forms.formsets import DELETION_FIELD_NAME, all_valid
from django.forms.models import (
from django.forms.widgets import CheckboxSelectMultiple, SelectMultiple
from django.http import HttpResponseRedirect
from django.http.response import HttpResponseBase
from django.template.response import SimpleTemplateResponse, TemplateResponse
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.utils.html import format_html
from django.utils.http import urlencode
from django.utils.safestring import mark_safe
from django.utils.text import (
from django.utils.translation import gettext as _
from django.utils.translation import ngettext
from django.views.decorators.csrf import csrf_protect
from django.views.generic import RedirectView
class InlineModelAdmin(BaseModelAdmin):
    """
    Options for inline editing of ``model`` instances.

    Provide ``fk_name`` to specify the attribute name of the ``ForeignKey``
    from ``model`` to its parent. This is required if ``model`` has more than
    one ``ForeignKey`` to its parent.
    """
    model = None
    fk_name = None
    formset = BaseInlineFormSet
    extra = 3
    min_num = None
    max_num = None
    template = None
    verbose_name = None
    verbose_name_plural = None
    can_delete = True
    show_change_link = False
    checks_class = InlineModelAdminChecks
    classes = None

    def __init__(self, parent_model, admin_site):
        self.admin_site = admin_site
        self.parent_model = parent_model
        self.opts = self.model._meta
        self.has_registered_model = admin_site.is_registered(self.model)
        super().__init__()
        if self.verbose_name_plural is None:
            if self.verbose_name is None:
                self.verbose_name_plural = self.opts.verbose_name_plural
            else:
                self.verbose_name_plural = format_lazy('{}s', self.verbose_name)
        if self.verbose_name is None:
            self.verbose_name = self.opts.verbose_name

    @property
    def media(self):
        extra = '' if settings.DEBUG else '.min'
        js = ['vendor/jquery/jquery%s.js' % extra, 'jquery.init.js', 'inlines.js']
        if self.filter_vertical or self.filter_horizontal:
            js.extend(['SelectBox.js', 'SelectFilter2.js'])
        if self.classes and 'collapse' in self.classes:
            js.append('collapse.js')
        return forms.Media(js=['admin/js/%s' % url for url in js])

    def get_extra(self, request, obj=None, **kwargs):
        """Hook for customizing the number of extra inline forms."""
        return self.extra

    def get_min_num(self, request, obj=None, **kwargs):
        """Hook for customizing the min number of inline forms."""
        return self.min_num

    def get_max_num(self, request, obj=None, **kwargs):
        """Hook for customizing the max number of extra inline forms."""
        return self.max_num

    def get_formset(self, request, obj=None, **kwargs):
        """Return a BaseInlineFormSet class for use in admin add/change views."""
        if 'fields' in kwargs:
            fields = kwargs.pop('fields')
        else:
            fields = flatten_fieldsets(self.get_fieldsets(request, obj))
        excluded = self.get_exclude(request, obj)
        exclude = [] if excluded is None else list(excluded)
        exclude.extend(self.get_readonly_fields(request, obj))
        if excluded is None and hasattr(self.form, '_meta') and self.form._meta.exclude:
            exclude.extend(self.form._meta.exclude)
        exclude = exclude or None
        can_delete = self.can_delete and self.has_delete_permission(request, obj)
        defaults = {'form': self.form, 'formset': self.formset, 'fk_name': self.fk_name, 'fields': fields, 'exclude': exclude, 'formfield_callback': partial(self.formfield_for_dbfield, request=request), 'extra': self.get_extra(request, obj, **kwargs), 'min_num': self.get_min_num(request, obj, **kwargs), 'max_num': self.get_max_num(request, obj, **kwargs), 'can_delete': can_delete, **kwargs}
        base_model_form = defaults['form']
        can_change = self.has_change_permission(request, obj) if request else True
        can_add = self.has_add_permission(request, obj) if request else True

        class DeleteProtectedModelForm(base_model_form):

            def hand_clean_DELETE(self):
                """
                We don't validate the 'DELETE' field itself because on
                templates it's not rendered using the field information, but
                just using a generic "deletion_field" of the InlineModelAdmin.
                """
                if self.cleaned_data.get(DELETION_FIELD_NAME, False):
                    using = router.db_for_write(self._meta.model)
                    collector = NestedObjects(using=using)
                    if self.instance._state.adding:
                        return
                    collector.collect([self.instance])
                    if collector.protected:
                        objs = []
                        for p in collector.protected:
                            objs.append(_('%(class_name)s %(instance)s') % {'class_name': p._meta.verbose_name, 'instance': p})
                        params = {'class_name': self._meta.model._meta.verbose_name, 'instance': self.instance, 'related_objects': get_text_list(objs, _('and'))}
                        msg = _('Deleting %(class_name)s %(instance)s would require deleting the following protected related objects: %(related_objects)s')
                        raise ValidationError(msg, code='deleting_protected', params=params)

            def is_valid(self):
                result = super().is_valid()
                self.hand_clean_DELETE()
                return result

            def has_changed(self):
                if not can_change and (not self.instance._state.adding):
                    return False
                if not can_add and self.instance._state.adding:
                    return False
                return super().has_changed()
        defaults['form'] = DeleteProtectedModelForm
        if defaults['fields'] is None and (not modelform_defines_fields(defaults['form'])):
            defaults['fields'] = forms.ALL_FIELDS
        return inlineformset_factory(self.parent_model, self.model, **defaults)

    def _get_form_for_get_fields(self, request, obj=None):
        return self.get_formset(request, obj, fields=None).form

    def get_queryset(self, request):
        queryset = super().get_queryset(request)
        if not self.has_view_or_change_permission(request):
            queryset = queryset.none()
        return queryset

    def _has_any_perms_for_target_model(self, request, perms):
        """
        This method is called only when the ModelAdmin's model is for an
        ManyToManyField's implicit through model (if self.opts.auto_created).
        Return True if the user has any of the given permissions ('add',
        'change', etc.) for the model that points to the through model.
        """
        opts = self.opts
        for field in opts.fields:
            if field.remote_field and field.remote_field.model != self.parent_model:
                opts = field.remote_field.model._meta
                break
        return any((request.user.has_perm('%s.%s' % (opts.app_label, get_permission_codename(perm, opts))) for perm in perms))

    def has_add_permission(self, request, obj):
        if self.opts.auto_created:
            return self._has_any_perms_for_target_model(request, ['change'])
        return super().has_add_permission(request)

    def has_change_permission(self, request, obj=None):
        if self.opts.auto_created:
            return self._has_any_perms_for_target_model(request, ['change'])
        return super().has_change_permission(request)

    def has_delete_permission(self, request, obj=None):
        if self.opts.auto_created:
            return self._has_any_perms_for_target_model(request, ['change'])
        return super().has_delete_permission(request, obj)

    def has_view_permission(self, request, obj=None):
        if self.opts.auto_created:
            return self._has_any_perms_for_target_model(request, ['view', 'change'])
        return super().has_view_permission(request)