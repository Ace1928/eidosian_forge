import json
from django import forms
from django.contrib.admin.utils import (
from django.core.exceptions import ObjectDoesNotExist
from django.db.models.fields.related import (
from django.forms.utils import flatatt
from django.template.defaultfilters import capfirst, linebreaksbr
from django.urls import NoReverseMatch, reverse
from django.utils.html import conditional_escape, format_html
from django.utils.safestring import mark_safe
from django.utils.translation import gettext
from django.utils.translation import gettext_lazy as _
class AdminReadonlyField:

    def __init__(self, form, field, is_first, model_admin=None):
        if callable(field):
            class_name = field.__name__ if field.__name__ != '<lambda>' else ''
        else:
            class_name = field
        if form._meta.labels and class_name in form._meta.labels:
            label = form._meta.labels[class_name]
        else:
            label = label_for_field(field, form._meta.model, model_admin, form=form)
        if form._meta.help_texts and class_name in form._meta.help_texts:
            help_text = form._meta.help_texts[class_name]
        else:
            help_text = help_text_for_field(class_name, form._meta.model)
        if field in form.fields:
            is_hidden = form.fields[field].widget.is_hidden
        else:
            is_hidden = False
        self.field = {'name': class_name, 'label': label, 'help_text': help_text, 'field': field, 'is_hidden': is_hidden}
        self.form = form
        self.model_admin = model_admin
        self.is_first = is_first
        self.is_checkbox = False
        self.is_readonly = True
        self.empty_value_display = model_admin.get_empty_value_display()

    def label_tag(self):
        attrs = {}
        if not self.is_first:
            attrs['class'] = 'inline'
        label = self.field['label']
        return format_html('<label{}>{}{}</label>', flatatt(attrs), capfirst(label), self.form.label_suffix)

    def get_admin_url(self, remote_field, remote_obj):
        url_name = 'admin:%s_%s_change' % (remote_field.model._meta.app_label, remote_field.model._meta.model_name)
        try:
            url = reverse(url_name, args=[quote(remote_obj.pk)], current_app=self.model_admin.admin_site.name)
            return format_html('<a href="{}">{}</a>', url, remote_obj)
        except NoReverseMatch:
            return str(remote_obj)

    def contents(self):
        from django.contrib.admin.templatetags.admin_list import _boolean_icon
        field, obj, model_admin = (self.field['field'], self.form.instance, self.model_admin)
        try:
            f, attr, value = lookup_field(field, obj, model_admin)
        except (AttributeError, ValueError, ObjectDoesNotExist):
            result_repr = self.empty_value_display
        else:
            if field in self.form.fields:
                widget = self.form[field].field.widget
                if getattr(widget, 'read_only', False):
                    return widget.render(field, value)
            if f is None:
                if getattr(attr, 'boolean', False):
                    result_repr = _boolean_icon(value)
                elif hasattr(value, '__html__'):
                    result_repr = value
                else:
                    result_repr = linebreaksbr(value)
            else:
                if isinstance(f.remote_field, ManyToManyRel) and value is not None:
                    result_repr = ', '.join(map(str, value.all()))
                elif isinstance(f.remote_field, (ForeignObjectRel, OneToOneField)) and value is not None:
                    result_repr = self.get_admin_url(f.remote_field, value)
                else:
                    result_repr = display_for_field(value, f, self.empty_value_display)
                result_repr = linebreaksbr(result_repr)
        return conditional_escape(result_repr)