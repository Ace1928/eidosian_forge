from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_request('stepBack')
@register
class StepBackRequest(BaseSchema):
    """
    The request executes one backward step (in the given granularity) for the specified thread and
    allows all other threads to run backward freely by resuming them.
    
    If the debug adapter supports single thread execution (see capability
    'supportsSingleThreadExecutionRequests') setting the 'singleThread' argument to true prevents other
    suspended threads from resuming.
    
    The debug adapter first sends the response and then a 'stopped' event (with reason 'step') after the
    step has completed.
    
    Clients should only call this request if the capability 'supportsStepBack' is true.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['request']}, 'command': {'type': 'string', 'enum': ['stepBack']}, 'arguments': {'type': 'StepBackArguments'}}
    __refs__ = set(['arguments'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, arguments, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string command: 
        :param StepBackArguments arguments: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'request'
        self.command = 'stepBack'
        if arguments is None:
            self.arguments = StepBackArguments()
        else:
            self.arguments = StepBackArguments(update_ids_from_dap=update_ids_from_dap, **arguments) if arguments.__class__ != StepBackArguments else arguments
        self.seq = seq
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        type = self.type
        command = self.command
        arguments = self.arguments
        seq = self.seq
        dct = {'type': type, 'command': command, 'arguments': arguments.to_dict(update_ids_to_dap=update_ids_to_dap), 'seq': seq}
        dct.update(self.kwargs)
        return dct