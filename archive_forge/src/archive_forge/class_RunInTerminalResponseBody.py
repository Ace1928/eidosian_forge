from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class RunInTerminalResponseBody(BaseSchema):
    """
    "body" of RunInTerminalResponse

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'processId': {'type': 'integer', 'description': 'The process ID. The value should be less than or equal to 2147483647 (2^31-1).'}, 'shellProcessId': {'type': 'integer', 'description': 'The process ID of the terminal shell. The value should be less than or equal to 2147483647 (2^31-1).'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, processId=None, shellProcessId=None, update_ids_from_dap=False, **kwargs):
        """
        :param integer processId: The process ID. The value should be less than or equal to 2147483647 (2^31-1).
        :param integer shellProcessId: The process ID of the terminal shell. The value should be less than or equal to 2147483647 (2^31-1).
        """
        self.processId = processId
        self.shellProcessId = shellProcessId
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        processId = self.processId
        shellProcessId = self.shellProcessId
        dct = {}
        if processId is not None:
            dct['processId'] = processId
        if shellProcessId is not None:
            dct['shellProcessId'] = shellProcessId
        dct.update(self.kwargs)
        return dct