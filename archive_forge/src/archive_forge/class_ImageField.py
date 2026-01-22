import datetime
import posixpath
from django import forms
from django.core import checks
from django.core.files.base import File
from django.core.files.images import ImageFile
from django.core.files.storage import Storage, default_storage
from django.core.files.utils import validate_file_name
from django.db.models import signals
from django.db.models.fields import Field
from django.db.models.query_utils import DeferredAttribute
from django.db.models.utils import AltersData
from django.utils.translation import gettext_lazy as _
class ImageField(FileField):
    attr_class = ImageFieldFile
    descriptor_class = ImageFileDescriptor
    description = _('Image')

    def __init__(self, verbose_name=None, name=None, width_field=None, height_field=None, **kwargs):
        self.width_field, self.height_field = (width_field, height_field)
        super().__init__(verbose_name, name, **kwargs)

    def check(self, **kwargs):
        return [*super().check(**kwargs), *self._check_image_library_installed()]

    def _check_image_library_installed(self):
        try:
            from PIL import Image
        except ImportError:
            return [checks.Error('Cannot use ImageField because Pillow is not installed.', hint='Get Pillow at https://pypi.org/project/Pillow/ or run command "python -m pip install Pillow".', obj=self, id='fields.E210')]
        else:
            return []

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.width_field:
            kwargs['width_field'] = self.width_field
        if self.height_field:
            kwargs['height_field'] = self.height_field
        return (name, path, args, kwargs)

    def contribute_to_class(self, cls, name, **kwargs):
        super().contribute_to_class(cls, name, **kwargs)
        if not cls._meta.abstract and (self.width_field or self.height_field):
            signals.post_init.connect(self.update_dimension_fields, sender=cls)

    def update_dimension_fields(self, instance, force=False, *args, **kwargs):
        """
        Update field's width and height fields, if defined.

        This method is hooked up to model's post_init signal to update
        dimensions after instantiating a model instance.  However, dimensions
        won't be updated if the dimensions fields are already populated.  This
        avoids unnecessary recalculation when loading an object from the
        database.

        Dimensions can be forced to update with force=True, which is how
        ImageFileDescriptor.__set__ calls this method.
        """
        has_dimension_fields = self.width_field or self.height_field
        if not has_dimension_fields or self.attname not in instance.__dict__:
            return
        file = getattr(instance, self.attname)
        if not file and (not force):
            return
        dimension_fields_filled = not (self.width_field and (not getattr(instance, self.width_field)) or (self.height_field and (not getattr(instance, self.height_field))))
        if dimension_fields_filled and (not force):
            return
        if file:
            width = file.width
            height = file.height
        else:
            width = None
            height = None
        if self.width_field:
            setattr(instance, self.width_field, width)
        if self.height_field:
            setattr(instance, self.height_field, height)

    def formfield(self, **kwargs):
        return super().formfield(**{'form_class': forms.ImageField, **kwargs})