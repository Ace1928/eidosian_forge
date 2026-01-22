import logging
from os_ken.services.protocols.bgp.operator.views import fields
def _create_collection_view(detail_view_class, name, encode=None, view_class=None):
    assert issubclass(detail_view_class, OperatorDetailView)
    class_fields = detail_view_class._collect_fields()
    if encode is not None:
        class_fields.update({'encode': encode})
    return type(name, (view_class,), class_fields)