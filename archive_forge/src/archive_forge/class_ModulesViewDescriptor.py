from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ModulesViewDescriptor(BaseSchema):
    """
    The ModulesViewDescriptor is the container for all declarative configuration options of a
    ModuleView.
    
    For now it only specifies the columns to be shown in the modules view.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'columns': {'type': 'array', 'items': {'$ref': '#/definitions/ColumnDescriptor'}}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, columns, update_ids_from_dap=False, **kwargs):
        """
        :param array columns: 
        """
        self.columns = columns
        if update_ids_from_dap and self.columns:
            for o in self.columns:
                ColumnDescriptor.update_dict_ids_from_dap(o)
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        columns = self.columns
        if columns and hasattr(columns[0], 'to_dict'):
            columns = [x.to_dict() for x in columns]
        dct = {'columns': [ColumnDescriptor.update_dict_ids_to_dap(o) for o in columns] if update_ids_to_dap and columns else columns}
        dct.update(self.kwargs)
        return dct