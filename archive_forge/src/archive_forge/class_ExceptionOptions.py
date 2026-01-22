from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ExceptionOptions(BaseSchema):
    """
    An ExceptionOptions assigns configuration options to a set of exceptions.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'path': {'type': 'array', 'items': {'$ref': '#/definitions/ExceptionPathSegment'}, 'description': "A path that selects a single or multiple exceptions in a tree. If 'path' is missing, the whole tree is selected.\nBy convention the first segment of the path is a category that is used to group exceptions in the UI."}, 'breakMode': {'description': 'Condition when a thrown exception should result in a break.', 'type': 'ExceptionBreakMode'}}
    __refs__ = set(['breakMode'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, breakMode, path=None, update_ids_from_dap=False, **kwargs):
        """
        :param ExceptionBreakMode breakMode: Condition when a thrown exception should result in a break.
        :param array path: A path that selects a single or multiple exceptions in a tree. If 'path' is missing, the whole tree is selected.
        By convention the first segment of the path is a category that is used to group exceptions in the UI.
        """
        if breakMode is not None:
            assert breakMode in ExceptionBreakMode.VALID_VALUES
        self.breakMode = breakMode
        self.path = path
        if update_ids_from_dap and self.path:
            for o in self.path:
                ExceptionPathSegment.update_dict_ids_from_dap(o)
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        breakMode = self.breakMode
        path = self.path
        if path and hasattr(path[0], 'to_dict'):
            path = [x.to_dict() for x in path]
        dct = {'breakMode': breakMode}
        if path is not None:
            dct['path'] = [ExceptionPathSegment.update_dict_ids_to_dap(o) for o in path] if update_ids_to_dap and path else path
        dct.update(self.kwargs)
        return dct