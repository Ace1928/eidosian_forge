from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class TerminateArguments(BaseSchema):
    """
    Arguments for 'terminate' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'restart': {'type': 'boolean', 'description': "A value of true indicates that this 'terminate' request is part of a restart sequence."}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, restart=None, update_ids_from_dap=False, **kwargs):
        """
        :param boolean restart: A value of true indicates that this 'terminate' request is part of a restart sequence.
        """
        self.restart = restart
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        restart = self.restart
        dct = {}
        if restart is not None:
            dct['restart'] = restart
        dct.update(self.kwargs)
        return dct