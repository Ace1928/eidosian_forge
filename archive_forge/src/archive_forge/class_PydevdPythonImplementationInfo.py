from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class PydevdPythonImplementationInfo(BaseSchema):
    """
    This object contains python implementation details.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'name': {'type': 'string', 'description': 'Python implementation name.'}, 'version': {'type': 'string', 'description': 'Python version as a string in semver format: <major>.<minor>.<micro><releaselevel><serial>.'}, 'description': {'type': 'string', 'description': 'Optional description for this python implementation.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, name=None, version=None, description=None, update_ids_from_dap=False, **kwargs):
        """
        :param string name: Python implementation name.
        :param string version: Python version as a string in semver format: <major>.<minor>.<micro><releaselevel><serial>.
        :param string description: Optional description for this python implementation.
        """
        self.name = name
        self.version = version
        self.description = description
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        name = self.name
        version = self.version
        description = self.description
        dct = {}
        if name is not None:
            dct['name'] = name
        if version is not None:
            dct['version'] = version
        if description is not None:
            dct['description'] = description
        dct.update(self.kwargs)
        return dct