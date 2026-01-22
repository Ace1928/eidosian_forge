import json
from xml.dom import pulldom
from xml.sax import handler
from xml.sax.expatreader import ExpatParser as _ExpatParser
from django.apps import apps
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.core.serializers import base
from django.db import DEFAULT_DB_ALIAS, models
from django.utils.xmlutils import SimplerXMLGenerator, UnserializableContentError
def _handle_m2m_field_node(self, node, field):
    """
        Handle a <field> node for a ManyToManyField.
        """
    model = field.remote_field.model
    default_manager = model._default_manager
    if hasattr(default_manager, 'get_by_natural_key'):

        def m2m_convert(n):
            keys = n.getElementsByTagName('natural')
            if keys:
                field_value = [getInnerText(k).strip() for k in keys]
                obj_pk = default_manager.db_manager(self.db).get_by_natural_key(*field_value).pk
            else:
                obj_pk = model._meta.pk.to_python(n.getAttribute('pk'))
            return obj_pk
    else:

        def m2m_convert(n):
            return model._meta.pk.to_python(n.getAttribute('pk'))
    values = []
    try:
        for c in node.getElementsByTagName('object'):
            values.append(m2m_convert(c))
    except Exception as e:
        if isinstance(e, ObjectDoesNotExist) and self.handle_forward_references:
            return base.DEFER_FIELD
        else:
            raise base.M2MDeserializationError(e, c)
    else:
        return values