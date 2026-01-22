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
def _handle_fk_field_node(self, node, field):
    """
        Handle a <field> node for a ForeignKey
        """
    if node.getElementsByTagName('None'):
        return None
    else:
        model = field.remote_field.model
        if hasattr(model._default_manager, 'get_by_natural_key'):
            keys = node.getElementsByTagName('natural')
            if keys:
                field_value = [getInnerText(k).strip() for k in keys]
                try:
                    obj = model._default_manager.db_manager(self.db).get_by_natural_key(*field_value)
                except ObjectDoesNotExist:
                    if self.handle_forward_references:
                        return base.DEFER_FIELD
                    else:
                        raise
                obj_pk = getattr(obj, field.remote_field.field_name)
                if field.remote_field.model._meta.pk.remote_field:
                    obj_pk = obj_pk.pk
            else:
                field_value = getInnerText(node).strip()
                obj_pk = model._meta.get_field(field.remote_field.field_name).to_python(field_value)
            return obj_pk
        else:
            field_value = getInnerText(node).strip()
            return model._meta.get_field(field.remote_field.field_name).to_python(field_value)