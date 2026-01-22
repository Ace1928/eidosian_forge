from io import StringIO
from django.core.exceptions import ObjectDoesNotExist
from django.db import models
class DeserializedObject:
    """
    A deserialized model.

    Basically a container for holding the pre-saved deserialized data along
    with the many-to-many data saved with the object.

    Call ``save()`` to save the object (with the many-to-many data) to the
    database; call ``save(save_m2m=False)`` to save just the object fields
    (and not touch the many-to-many stuff.)
    """

    def __init__(self, obj, m2m_data=None, deferred_fields=None):
        self.object = obj
        self.m2m_data = m2m_data
        self.deferred_fields = deferred_fields

    def __repr__(self):
        return '<%s: %s(pk=%s)>' % (self.__class__.__name__, self.object._meta.label, self.object.pk)

    def save(self, save_m2m=True, using=None, **kwargs):
        models.Model.save_base(self.object, using=using, raw=True, **kwargs)
        if self.m2m_data and save_m2m:
            for accessor_name, object_list in self.m2m_data.items():
                getattr(self.object, accessor_name).set(object_list)
        self.m2m_data = None

    def save_deferred_fields(self, using=None):
        self.m2m_data = {}
        for field, field_value in self.deferred_fields.items():
            opts = self.object._meta
            label = opts.app_label + '.' + opts.model_name
            if isinstance(field.remote_field, models.ManyToManyRel):
                try:
                    values = deserialize_m2m_values(field, field_value, using, handle_forward_references=False)
                except M2MDeserializationError as e:
                    raise DeserializationError.WithData(e.original_exc, label, self.object.pk, e.pk)
                self.m2m_data[field.name] = values
            elif isinstance(field.remote_field, models.ManyToOneRel):
                try:
                    value = deserialize_fk_value(field, field_value, using, handle_forward_references=False)
                except Exception as e:
                    raise DeserializationError.WithData(e, label, self.object.pk, field_value)
                setattr(self.object, field.attname, value)
        self.save()