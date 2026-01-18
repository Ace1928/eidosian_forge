from datetime import datetime
from wsme import types as wsme_types
from glance.common import timeutils
@classmethod
def to_wsme_model(model, db_entity, self_link=None, schema=None):
    names = []
    for attribute in model._wsme_attributes:
        names.append(attribute.name)
    values = {}
    for name in names:
        value = getattr(db_entity, name, None)
        if value is not None:
            if isinstance(value, datetime):
                iso_datetime_value = timeutils.isotime(value)
                values.update({name: iso_datetime_value})
            else:
                values.update({name: value})
    if schema:
        values['schema'] = schema
    model_object = model(**values)
    if self_link:
        model_object.self = self_link
    return model_object