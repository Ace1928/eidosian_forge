import copy
import json
from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import URLValidator
from django.db.models import CASCADE, UUIDField
from django.urls import reverse
from django.urls.exceptions import NoReverseMatch
from django.utils.html import smart_urlquote
from django.utils.http import urlencode
from django.utils.text import Truncator
from django.utils.translation import get_language
from django.utils.translation import gettext as _
class ForeignKeyRawIdWidget(forms.TextInput):
    """
    A Widget for displaying ForeignKeys in the "raw_id" interface rather than
    in a <select> box.
    """
    template_name = 'admin/widgets/foreign_key_raw_id.html'

    def __init__(self, rel, admin_site, attrs=None, using=None):
        self.rel = rel
        self.admin_site = admin_site
        self.db = using
        super().__init__(attrs)

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        rel_to = self.rel.model
        if self.admin_site.is_registered(rel_to):
            related_url = reverse('admin:%s_%s_changelist' % (rel_to._meta.app_label, rel_to._meta.model_name), current_app=self.admin_site.name)
            params = self.url_parameters()
            if params:
                related_url += '?' + urlencode(params)
            context['related_url'] = related_url
            context['link_title'] = _('Lookup')
            css_class = 'vForeignKeyRawIdAdminField'
            if isinstance(self.rel.get_related_field(), UUIDField):
                css_class += ' vUUIDField'
            context['widget']['attrs'].setdefault('class', css_class)
        else:
            context['related_url'] = None
        if context['widget']['value']:
            context['link_label'], context['link_url'] = self.label_and_url_for_value(value)
        else:
            context['link_label'] = None
        return context

    def base_url_parameters(self):
        limit_choices_to = self.rel.limit_choices_to
        if callable(limit_choices_to):
            limit_choices_to = limit_choices_to()
        return url_params_from_lookup_dict(limit_choices_to)

    def url_parameters(self):
        from django.contrib.admin.views.main import TO_FIELD_VAR
        params = self.base_url_parameters()
        params.update({TO_FIELD_VAR: self.rel.get_related_field().name})
        return params

    def label_and_url_for_value(self, value):
        key = self.rel.get_related_field().name
        try:
            obj = self.rel.model._default_manager.using(self.db).get(**{key: value})
        except (ValueError, self.rel.model.DoesNotExist, ValidationError):
            return ('', '')
        try:
            url = reverse('%s:%s_%s_change' % (self.admin_site.name, obj._meta.app_label, obj._meta.object_name.lower()), args=(obj.pk,))
        except NoReverseMatch:
            url = ''
        return (Truncator(obj).words(14), url)