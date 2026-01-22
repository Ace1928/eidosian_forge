from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ModulesResponseBody(BaseSchema):
    """
    "body" of ModulesResponse

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'modules': {'type': 'array', 'items': {'$ref': '#/definitions/Module'}, 'description': 'All modules or range of modules.'}, 'totalModules': {'type': 'integer', 'description': 'The total number of modules available.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, modules, totalModules=None, update_ids_from_dap=False, **kwargs):
        """
        :param array modules: All modules or range of modules.
        :param integer totalModules: The total number of modules available.
        """
        self.modules = modules
        if update_ids_from_dap and self.modules:
            for o in self.modules:
                Module.update_dict_ids_from_dap(o)
        self.totalModules = totalModules
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        modules = self.modules
        if modules and hasattr(modules[0], 'to_dict'):
            modules = [x.to_dict() for x in modules]
        totalModules = self.totalModules
        dct = {'modules': [Module.update_dict_ids_to_dap(o) for o in modules] if update_ids_to_dap and modules else modules}
        if totalModules is not None:
            dct['totalModules'] = totalModules
        dct.update(self.kwargs)
        return dct